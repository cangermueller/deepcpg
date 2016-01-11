query_db <- function(table) {
  con <- src_sqlite(opts$db_file)
  d <- tbl(con, sql(sprintf('SELECT * FROM %s', table)))
  d <- d %>% collect
  if ('target' %in% names(d)) {
    d <- d %>% mutate(cell_type=parse_cell_type(target))
  }
  d <- d %>% char_to_factor %>% droplevels %>% tbl_df

  # d <- d %>% filter(seqmut == opts$seqmut) %>% select(-seqmut)
  d <- d %>% mutate(fun=paste0(fun, '_')) %>% spread(fun, value)
  if (opts$norm_effect) {
    d <- d %>% group_by(seqmut, effect) %>% mutate(mean_=mean_/max(mean_))
    if ('mean0_' %in% names(d)) {
      d <- d %>% group_by(effect) %>% mutate(mean0_=mean0_/max(mean0_))
    }
  }
  return (d)
}

query_stats <- function(...) {
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

plot_global <- function(d) {
  p <- ggplot(d, aes(x=effect, y=mean_)) +
    geom_boxplot(aes(fill=effect), outlier.size=0) +
    geom_jitter(aes(color=cell_type), position=position_jitter(width=0.1, height=0)) +
    scale_color_manual(values=colors_$cell_type) +
    xlab('') + ylab('') +
    theme_pub() +
    theme(legend.position='right') +
    theme(axis.text.x=element_text(angle=40, hjust=1))
  return (p)
}

plot_annos <- function(d, annos) {
  d <- d %>% mutate(anno=factor(anno, levels=annos))
  p <- ggplot(d, aes(x=effect, y=mean_)) +
    geom_boxplot(aes(fill=effect), outlier.size=0) +
    geom_point(aes(fill=effect, color=cell_type), size=0.5,
      position=position_jitterdodge(jitter.width=0, jitter.height=0, dodge.width=0.8)) +
    scale_color_manual(values=colors_$cell_type) +
    theme_pub() +
    xlab('') + ylab('Effect') +
    facet_wrap(~anno)
  return (p)
}

plot_annos_t <- function(d, annos) {
  d <- d %>%
    mutate(anno=factor(anno, levels=annos)) %>%
    gather(group, value, one_of('mean_', 'mean0_')) %>%
    mutate(group=factor(group, levels=c('mean_', 'mean0_'), labels=c('inside', 'outside'))) %>% droplevels
  p <- ggplot(d, aes(x=effect, y=value)) +
    geom_boxplot(aes(fill=group), outlier.size=0) +
    geom_point(
      aes(fill=group, shape=cell_type, color=-log10(pvalue)), size=0.9,
      position=position_jitterdodge(jitter.width=0.3, jitter.height=0, dodge.width=0.8)) +
    scale_color_gradient2(low='black', mid='red', high='red', midpoint=200) +
    theme_pub() +
    xlab('') + ylab('Effect') +
    facet_wrap(~anno, ncol=3)
  return (p)
}

plot_stats <- function(d) {
  p <- ggplot(d, aes(x=bin_mid, y=mean_)) +
    geom_smooth(se=F, size=1.2, aes(color=effect, fill=stat)) +
    geom_point(size=0.7, aes(color=effect, shape=cell_type),
      position=position_jitter(width=0.005, height=0)) +
    xlab('') + ylab('Effect') +
    facet_wrap(~stat, ncol=1, scales='free') +
    theme_pub() +
    theme(legend.position='right') +
    guides(fill=F)
  return (p)
}

