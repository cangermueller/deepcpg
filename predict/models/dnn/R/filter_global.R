query_db <- function(db_file, effect, table='global', cond=NULL) {
  con <- src_sqlite(db_file)
  h <- sprintf('SELECT * FROM %s WHERE effect = "%s"', table, effect)
  if (!is.null(cond)) {
    h <- sprintf('%s AND %s', h, cond)
  }
  d <- tbl(con, sql(h))
  d <- d %>% collect %>% select(-path, -id, -effect)
  d <- d %>% mutate(cell_type=parse_cell_type(target))
  d <- d %>% rename(filt=seqmut) %>%
    mutate(filt=factor(as.numeric(gsub('^z_f', '', filt))))
  d <- d %>% mutate(fun=paste0(fun, '_')) %>% spread(fun, value)
  d <- d %>% group_by(target) %>% mutate(rank_=rank(-abs(mean_))) %>%
    ungroup
  d <- d %>% char_to_factor %>% droplevels %>% tbl_df %>%
    move_cols(c('filt', 'target', 'cell_type'))
  return (d)
}

query_stats <- function(db_file, effect) {
  d <- query_db(db_file, effect, table='stats')
  d <- d %>%
    mutate(bin=gsub('\\[', '', gsub('\\]', '', gsub('[()]', '', bin)))) %>%
    separate(bin, c('bin_lo', 'bin_up'), ', ') %>%
    mutate(bin_lo=as.numeric(bin_lo), bin_up=as.numeric(bin_up),
        bin_mid=0.5*(bin_lo + bin_up))
  h <- sub('win_(.+)', '\\1 (win)', d$stat)
  hh <- move_front(sort(unique(h)), c('cg_obs_exp', 'gc_content'))
  d <- d %>% mutate(stat=factor(h, levels=hh))
  return (d)
}

plot_global <- function(d) {
  p <- ggplot(d, aes(x=label, y=mean_)) +
    geom_boxplot(aes(fill=cell_type), outlier.shape=NA) +
    geom_point(aes(fill=cell_type), size=0.8,
      position=position_jitterdodge(jitter.width=0.5, jitter.height=0, dodge.width=0.8)) +
    scale_fill_manual(values=colors_$cell_type) +
    guides(color=F) +
    xlab('') + ylab('') +
    theme_pub() +
    theme(
      axis.text.x=element_text(angle=30, hjust=1)
      )
  return (p)
}

plot_annos <- function(d, annos) {
  p <- ggplot(d, aes(x=filt, y=mean_)) +
    geom_boxplot(aes(fill=cell_type), outlier.size=0) +
    geom_point(aes(fill=cell_type), size=0.5,
      position=position_jitterdodge(jitter.width=0, jitter.height=0, dodge.width=0.8)) +
    scale_fill_manual(values=colors_$cell_type) +
    theme_pub() +
    theme(
      axis.text.x=element_text(angle=30, hjust=1),
      legend.position='top'
      ) +
    xlab('') + ylab('Effect') +
    facet_wrap(~anno, ncol=1, scale='free')
  return (p)
}

plot_stats <- function(d) {
  p <- ggplot(d, aes(x=bin_mid, y=mean_)) +
    geom_smooth(se=F, size=1.2, aes(color=label, fill=stat)) +
    geom_point(size=0.7, aes(color=label, shape=cell_type),
      position=position_jitter(width=0.005, height=0)) +
    xlab('') + ylab('Effect') +
    facet_wrap(~stat, ncol=1, scales='free') +
    theme_pub() +
    theme(legend.position='top') +
    guides(color=guide_legend(ncol=3), fill=F)
  return (p)
}
