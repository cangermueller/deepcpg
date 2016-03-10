read_group <- function(path, group, nb_sample=NULL) {
  d <- list()
  for (k in c('y', 'z')) {
    d[[k]] <- h5read(path, sprintf('%s/%s', group, k))
  }
  d <- as.data.frame(d) %>% tbl_df %>% mutate(name=basename(group))
  if (!is.null(nb_sample)) {
    d <- d %>% sample_n(min(nb_sample, nrow(d)))
  }
  return (d)
}

read_pred <- function(path, target, annos_regex=NULL, nb_sample=NULL) {
  d <- list()
  d[[length(d) + 1]] <- read_group(path,
    sprintf('/%s/global', target),
    nb_sample)
  annos <- h5ls(path, sprintf('/%s/annos', target))
  if (!is.null(annos_regex)) {
    annos <- grep(annos_regex, annos, value=T)
  }
  for (anno in annos) {
    d[[length(d) + 1]] <- read_group(path,
      sprintf('/%s/annos/%s', target, anno),
      nb_sample)
  }
  d <- do.call(rbind.data.frame, d) %>%
    mutate(name=factor(name)) %>% tbl_df
  return (d)
}

query_db <- function(db_file, table, models=NULL) {
  con <- src_sqlite(db_file)
  d <- tbl(con, sql(sprintf('SELECT * FROM %s', table)))
  d <- d %>% collect
  if (!is.null(models)) {
    d <- d %>% filter(model %in% models)
  }
  d <- d %>% select(-c(path, id)) %>%
    gather(fun, value, -one_of(c('anno', 'target', 'model')))
  d <- d %>% parse_var_target
  d <- d %>% char_to_factor %>% droplevels %>% tbl_df %>%
    move_cols(c('model', 'target', 'cell_type', 'wlen'))
  return (d)
}

stats <- function(d, metric_='var') {
  s <- d %>% filter(metric==metric_) %>%
    group_by(anno, fun) %>% summarise(value=mean(value)) %>% ungroup %>%
    spread(fun, value) %>%
    arrange(desc(mse))
  return (s)
}

plot_score <- function(d, metric_='var', funs=c('mad', 'rmse', 'rs')) {
  d <- d %>% filter(metric == metric_, fun %in% funs) %>% droplevels %>%
    mutate(fun=factor(fun, levels=funs))
  levels(d$anno) <- format_annos(levels(d$anno))
  p <- ggplot(d, aes(x=anno, y=value)) +
    geom_bar(fill='grey', stat='identity', position='dodge') +
    theme_pub() +
    theme(
      axis.text.x=element_text(angle=30, hjust=1),
      axis.title.y=element_text(size=rel(0.8)),
      legend.position='top'
    ) +
    facet_wrap(~fun, ncol=1, scale='free') +
    xlab('') + ylab('')
  return (p)
}

metric_ylab <- function(metric) {
  if (metric == 'var') {
    metric <- 'Variance'
  } else if (metric == 'met') {
    metric <- 'Mean methylation'
  }
  return (metric)
}

metric_ylim <- function(metric) {
  if (metric == 'var') {
    ylim <- c(0, 0.12)
  } else {
    ylim <- c(0, 1)
  }
  return (ylim)
}

plot_dist_match <- function(dyz, ds, metric_='var', score='rs') {
  dyz <- dyz %>% filter(metric==metric_) %>% droplevels
  ds <- ds %>% filter(metric==metric_, fun == score) %>% spread(fun, value)
  ds$score <- ds[[score]]
  if (score %in% gopts$lo_better) {
    ds$score <- -ds$score
  }
  d <- dyz %>% inner_join(ds) %>%
    mutate(anno=factor(anno, levels=levels(dyz$anno))) %>% droplevels
  d <- d %>% gather(key, value, y, z) %>%
    mutate(key=factor(key, levels=c('y', 'z'), labels=c('True', 'Predicted')))
  levels(d$anno) <- format_annos(levels(d$anno))
  ylim_ <- c(0, 1)
  if (metric == 'var') {
    ylim_ <- c(0, 0.12)
  }
  p <- ggplot(d, aes(x=anno, y=value)) +
    geom_boxplot(aes(fill=key, alpha=score), outlier.shape=NA) +
    scale_fill_manual(values=colors_$yz) +
    theme_pub() +
    theme(
      legend.position='top'
    ) +
    guides(alpha=F) +
    xlab('') + ylab(metric_ylab(metric)) +
    ylim(metric_ylim(metric)) +
    theme(axis.text.x=element_text(angle=30, hjust=1))
  return (p)
}

plot_scatter <- function(d, metric_='mean', nb_sample=1000) {
  d <- d %>% filter(metric==metric_) %>% droplevels %>%
    group_by(anno) %>% sample_n(nb_sample, replace=T)
  max_ <- max(d$y, d$z)
  p <- ggplot(d, aes(x=z, y=y)) +
    geom_abline(slope=1, linetype='dashed', color='black') +
    stat_density2d(color='grey') +
    geom_point(size=0.005 , color='royalblue') +
    xlim(0, max_) + ylim(0, max_) +
    facet_wrap(~anno, ncol=5) +
    xlab('Predicted') + ylab('Truth') +
    theme_pub() +
    theme(
      axis.text.x=element_text(size=rel(0.9)),
      axis.text.y=element_text(size=rel(0.9))
      )
  return (p)
}

eval_rank <- function(d, steps=seq(0, 0.5, length.out=100)) {
  d <- d %>% mutate(y=rank(y), z=rank(z)) %>% arrange(desc(z))
  met <- list(n=c(), p=c(), tpr=c())
  for (s in steps) {
    i <- nrow(d) * s
    d1 <- d[1:i,]
    d0 <- d[(i + 1):nrow(d),]
    thr <- d1$z[nrow(d1)]
    met$n <- c(met$n, nrow(d1))
    met$p <- c(met$p, nrow(d1) / nrow(d))
    met$tpr <- c(met$tpr, sum(d1$y >= thr) / nrow(d1))
  }
  met <- as.data.frame(met) %>% tbl_df
  return (met)
}

plot_rank <- function(d, what='tpr') {
  d$value <- d[[what]]
  p <- ggplot(d, aes(x=p, y=value)) +
    geom_abline(slope=1, linetype='dashed', color='darkgrey') +
    geom_line(aes(color=anno), size=1.2) +
    theme_pub() +
    xlab('% most variable sites predicted') +
    ylab(what) +
    facet_wrap(~cell_type, ncol=1) +
    theme(legend.position='right')
  return (p)
}
