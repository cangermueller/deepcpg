parse_wlen <- function(x) {
  wlen  <- as.numeric(gsub('z_[^0123456789-]+', '', x))
  wlen[wlen < 0] <- 9999
  return (wlen)
}

query_db <- function(db_file, table, effect='lor', seqmut='rnd') {
  con <- src_sqlite(db_file)
  query <- sprintf(
    'SELECT * FROM %s WHERE effect = "%s" AND seqmut LIKE "z_%s%%"',
    table, effect, seqmut)
  d <- tbl(con, sql(query))
  d <- d %>% collect %>% select(-c(path, id))
  d <- d %>%
    mutate(
      cell_type=parse_cell_type(target),
      wlen=factor(parse_wlen(seqmut))
      ) %>%
    select(-seqmut, -effect)
  d <- d %>% mutate(fun=paste0(fun, '_')) %>% spread(fun, value)
  d <- d %>% char_to_factor %>% droplevels %>% tbl_df
  return (d)
}

format_stats <- function(d) {
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

plot_global <- function(d) {
  p <- ggplot(d, aes(x=wlen, y=mean_)) +
    geom_boxplot(aes(fill=cell_type), outlier.size=0) +
    geom_point(aes(fill=cell_type),
      position=position_jitterdodge(dodge.width=0.8, jitter.width=0.1, jitter.height=0)) +
    scale_fill_manual(values=colors_$cell_type) +
    xlab('') + ylab('') +
    theme_pub() +
    theme(legend.position='right') +
    theme(axis.text.x=element_text(angle=40, hjust=1))
  return (p)
}

plot_annos <- function(d, annos=NULL) {
  if (!is.null(annos)) {
    d <- d %>% mutate(anno=factor(anno, levels=annos))
  }
  p <- ggplot(d, aes(x=wlen, y=mean_)) +
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
    facet_wrap(~anno, ncol=3, scale='free')
  return (p)
}

plot_annos_t <- function(d, annos) {
  d <- d %>%
    mutate(anno=factor(anno, levels=annos)) %>%
    gather(group, value, one_of('mean_', 'mean0_')) %>%
    mutate(group=factor(group, levels=c('mean_', 'mean0_'), labels=c('inside', 'outside'))) %>% droplevels
  p <- ggplot(d, aes(x=wlen, y=value)) +
    geom_boxplot(aes(fill=group), outlier.size=0) +
    geom_point(
      aes(fill=group, shape=cell_type, color=-log10(pvalue_)), size=0.9,
      position=position_jitterdodge(jitter.width=0.3, jitter.height=0, dodge.width=0.8)) +
    scale_color_gradient2(low='black', mid='red', high='red', midpoint=200) +
    theme_pub() +
    xlab('') + ylab('Effect') +
    facet_wrap(~anno, ncol=3)
  return (p)
}

plot_stats <- function(d) {
  p <- ggplot(d, aes(x=bin_mid, y=mean_)) +
    geom_smooth(se=F, size=1.2, aes(color=wlen, fill=stat)) +
    geom_point(size=0.7, aes(color=wlen, shape=cell_type),
      position=position_jitter(width=0.005, height=0)) +
    xlab('') + ylab('Effect') +
    facet_wrap(~stat, ncol=1, scales='free') +
    theme_pub() +
    theme(legend.position='top') +
    guides(fill=F)
  return (p)
}

