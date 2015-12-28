query_db <- function(table, models=NULL) {
  con <- src_sqlite(opts$db_file)
  d <- tbl(con, sql(sprintf('SELECT * FROM %s', table)))
  d <- d %>% collect
  if (!is.null(models)) {
    d <- d %>% filter(model %in% models)
  }
  if ('target' %in% names(d)) {
    d <- d %>% mutate(cell_type=parse_cell_type(target))
  }
  d <- d %>% char_to_factor %>% droplevels %>% tbl_df
  return (d)
}

top_n <- function(n, include=c('rf', 'win_avg')) {
  d <- as.vector(dat$gstats$model) %>% head(n)
  if (!is.null(include)) {
    i <- setdiff(include, d)
    h <- length(d) + length(i)
    if (h > n) {
      d <- d[1:(length(d) - (h - n))]
    }
    d <- c(d, i)
  }
  return (d)
}

prefix_cols <- function(d, pref, excl='model') {
  n <- names(d)
  h <- !(n %in% excl)
  n[h] <- paste(pref, n[h], sep='.')
  names(d) <- n
  return (d)
}

join_list <- function(d) {
  h <- d[[1]]
  if (length(h) > 1) {
    for (i in 2:length(h)) {
      h <- left_join(h, d[[i]])
    }
  }
  return (h)
}

query_hparams <- function(db_file, stats_global=NULL) {
  con <- src_sqlite(db_file)

  tabs <- tbl(con,
    sql('SELECT name FROM sqlite_master WHERE type = "table"')) %>%
    collect %>% unlist

  d <- list()
  d[[1]] <- tbl(con,
    sql('SELECT model, optimizer, optimizer_params FROM model')) %>% collect

  d[[2]] <- tbl(con,
    sql('SELECT model, nb_hidden, batch_norm FROM model_target')) %>%
    collect %>% prefix_cols('tar')
  if ('model_seq' %in% tabs) {
    d[[length(d) + 1]] <- tbl(con,
      sql('SELECT model, nb_hidden, nb_filter, filter_len, pool_len, drop_in, drop_out FROM model_seq')) %>%
      collect %>% prefix_cols('seq')
  }
  if ('model_cpg' %in% tabs) {
    d[[length(d) + 1]] <- tbl(con,
      sql('SELECT model, nb_hidden, nb_filter, filter_len, pool_len, drop_in, drop_out FROM model_cpg')) %>%
      collect %>% prefix_cols('cpg')
  }
  d <- join_list(d)

  if (!is.null(stats_global)) {
    p <- d %>%
      select(-c(optimizer, optimizer_params), -contains('batch_norm'))
    s <- stats_global %>% select(model, auc) %>%
      mutate(smodel=gsub('^dnn_', '', gsub('_r.*', '', model)))
    d <- s %>% inner_join(p, by=c('smodel'='model')) %>% select(-smodel)
  }
  return(d)
}


query_stats <- function(db_file, ...) {
  d <- query_db('stats', ...)
  d <- d %>%
    mutate(bin=gsub('\\[', '', gsub('\\]', '', gsub('[()]', '', bin)))) %>%
    separate(bin, c('bin_lo', 'bin_up'), ', ') %>%
    mutate(bin_lo=as.numeric(bin_lo), bin_up=as.numeric(bin_up),
        bin_mid=0.5*(bin_lo + bin_up))
  h <- d$stat
  h <- sub('win_(.+)', '\\1 (win)', h)
  d <- d %>% mutate(stat=factor(h, levels=sort(unique(h))))
  return (d)
}


stats_global <- function(d) {
  d <- d %>% group_by(model) %>%
    summarise_each(funs(mean), auc, acc, mcc, tpr, tnr) %>%
    ungroup %>%
    arrange(desc(auc))
  return (d)
}

plot_global <- function(d) {
  d <- d  %>% droplevels %>%
    arrange(desc(auc)) %>%
    mutate(model=factor(model, levels=unique(model)))
  d <- d %>% select(-c(path, id, trial, eval))
  d <- d %>% gather(metric, value, -c(model, target, cell_type))
  p <- ggplot(d, aes(x=model, y=value)) +
    geom_boxplot(aes(fill=model), alpha=1.0, outlier.size=0) +
    geom_jitter(aes(color=cell_type),
      position=position_jitter(width=0.1, height=0), size=0.8) +
    scale_color_manual(values=colors_$cell_type) +
    xlab('') + ylab('') +
    theme_pub() +
    theme(legend.position='right') +
    theme(axis.text.x=element_text(angle=40, hjust=1)) +
    facet_wrap(~metric, ncol=1, scales='free')
  return (p)
}

plot_global_scatter <- function(d) {
  d <- d %>% select(model, target, cell_type, auc)
  d <- d %>% spread(model, auc) %>%
    gather(model, auc, -c(target, cell_type, rf))
  p <- ggplot(d, aes(x=rf, y=auc)) +
    geom_abline(slope=1, color='darkgrey', linetype='dashed') +
    geom_point(aes(color=cell_type), size=0.8) +
    scale_color_manual(values=colors_$cell_type) +
    facet_wrap(~model, ncol=3, scales='free') +
    xlab('RF') + ylab('Model') +
    theme_pub()
  return (p)
}

plot_lc <- function(d) {
  d <- d %>% select(model, epoch, loss, val_loss) %>%
    gather(score, value, -c(model, epoch))
  p <- ggplot(d, aes(x=epoch, value)) +
    geom_line(aes(color=score)) +
    facet_wrap(~model, ncol=3, scales='free') +
    xlab('') + ylab('loss') +
    theme_pub()
  return (p)
}

plot_hparams <- function(d) {
  d <- d %>% gather(param, value, -c(auc, model))
  p <- ggplot(d, aes(x=value, y=auc)) +
    geom_point(aes(color=model)) +
    geom_smooth(size=1, method='lm') +
    facet_wrap(~param, scales='free', ncol=3) +
    coord_cartesian(ylim=c(0.7, max(d$auc))) +
    guides(color=F) +
    xlab('') + ylab('AUC') +
    theme_pub() +
    theme(axis.text.x=element_text(angle=30, hjust=1))
  return (p)
}

plot_annos <- function(d) {
  h <- d %>% group_by(anno) %>%
    summarise(auc=mean(auc)) %>% arrange(desc(auc)) %>% select(anno) %>% unlist
  d <- d %>% mutate(anno=factor(anno, levels=h))
  p <- ggplot(d, aes(x=model, y=auc, fill=model)) +
    geom_boxplot(outlier.size=0) +
    geom_point(aes(fill=model, color=cell_type), size=0.8,
      position=position_jitterdodge(jitter.width=0, jitter.height=0, dodge.width=0.8)) +
    facet_wrap(~anno, ncol=3) +
    theme_pub() +
    theme(
      panel.grid.major=element_line(colour="grey60", size=0.1, linetype='solid'),
      panel.grid.minor=element_line(colour="grey60", size=0.1, linetype='dotted'),
      axis.text.x=element_text(angle=30, hjust=1)
    ) +
    xlab('') + ylab('AUC')
  return (p)
}

plot_stats <- function(d) {
  p <- ggplot(d, aes(x=bin_mid, y=auc)) +
    geom_smooth(se=F, size=1.2, aes(color=model, fill=stat)) +
    geom_point(size=0.7, aes(color=model)) +
    xlab('') + ylab('AUC') +
    facet_wrap(~stat, ncol=1, scales='free') +
    theme_pub() +
    theme(legend.position='right') +
    guides(fill=F)
  return (p)
}

