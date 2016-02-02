parse_wlen <- function(x) {
  wlen  <- as.numeric(gsub('z_[^0123456789-]+', '', x))
  wlen[wlen < 0] <- 9999
  return (wlen)
}

query_db <- function(db_file, table='global', annos=NULL, effect='lor', seqmut='z_rnd-1', cond=c()) {
  con <- src_sqlite(db_file)
  cmd <- sprintf('SELECT * FROM %s WHERE seqmut = "%s"', table, seqmut)
  if (!is.null(annos)) {
    cond <- c(cond, sprintf('anno LIKE "%s"', annos))
  }
  if (length(cond) > 0) {
    cmd <- sprintf('%s AND %s', cmd, paste0(cond, collapse=' AND '))
  }
  d <- tbl(con, sql(cmd))
  d <- d %>% collect %>% select(-c(path, id))
  d <- d %>% mutate(
    cell_type=parse_cell_type(target)
    )
  # d <- d %>% spread(effect, value)
  d <- d %>% filter_('fun == "mean"') %>% select(-fun) %>%
    spread(effect, value)
  if (effect == 'lor') {
    d$value_del <- d$lor
    d$value_abs <- d$abs_lor
  } else {
    d$value_del <- d$del
    d$value_abs <- d$abs
  }
  d <- d %>% char_to_factor %>% droplevels %>% tbl_df# %>%
    # move_cols(c('filt', 'target', 'cell_type'))
  return (d)
}

format_stats <- function(d) {
  d <- d %>%
    mutate(bin=gsub('\\[', '', gsub('\\]', '', gsub('[()]', '', bin)))) %>%
    separate(bin, c('bin_lo', 'bin_up'), ', ') %>%
    mutate(bin_lo=as.numeric(bin_lo), bin_up=as.numeric(bin_up),
        bin_mid=0.5*(bin_lo + bin_up),
        x=round(bin_mid, 3))
  h <- sub('win_(.+)', '\\1 (win)', d$stat)
  hh <- move_front(sort(unique(h)), c('cg_obs_exp', 'gc_content'))
  d <- d %>% mutate(stat=factor(h, levels=hh))
  return (d)
}

plot_global <- function(d) {
  h <- d %>% arrange(desc(value_abs)) %>% select(target) %>% unlist
  d <- d %>% mutate(target=factor(target, levels=h))
  p <- ggplot(d, aes(x=target, y=value_abs)) +
    geom_bar(aes(fill=cell_type), stat='identity') +
    scale_fill_manual(values=colors_$cell_type) +
    guides(color=F) +
    xlab('') + ylab('Effect') +
    theme_pub() +
    theme(
      legend.position='top',
      axis.text.x=element_text(angle=30, hjust=1)
      )
  return (p)
}

plot_annos <- function(d) {
  h <- d %>% group_by(anno) %>% summarise(value_abs=mean(value_abs)) %>%
    arrange(desc(value_abs)) %>% select(anno) %>% unlist
  d <- d %>% mutate(anno=factor(anno, levels=rev(h)))
  p <- ggplot(d, aes(x=anno, y=value_abs)) +
    geom_boxplot(aes(fill=cell_type), outlier.size=0) +
    geom_point(aes(fill=cell_type), size=0.5,
      position=position_jitterdodge(jitter.width=0, jitter.height=0, dodge.width=0.8)) +
    scale_fill_manual(values=colors_$cell_type) +
    theme_pub() +
    xlab('') + ylab('Effect') +
    coord_flip()
  return (p)
}

plot_annos_heat <- function(d, rev_colors=F, Rowv=T, Colv=T, lhei=NULL, del=F) {
  f <- ifelse(del, 'value_del', 'value_abs')
  d$value <- d[[f]]
  d <- d %>% select(anno, target, value) %>% spread(target, value) %>%
    as.data.frame
  rownames(d) <- d$anno
  d <- d %>% select(-anno) %>% as.matrix

  col_colors <- colors_$cell_type[parse_cell_type(colnames(d))]
  if (del) {
    colors <- rev(brewer.pal(11, 'RdBu'))
  } else {
    colors <- brewer.pal(9, 'YlGnBu')
  }
  colors <- colorRampPalette(colors)(50)
  dendro <- 'column'
  if (!is.null(Rowv)) {
    dendro <- 'both'
  }

  p <- heatmap.2(d, density.info='none', trace='none', col=colors,
    Rowv=Rowv, Colv=Colv, keysize=1.0, dendrogram=dendro,
    margins=c(8, 8), lhei=lhei, ColSideColors=col_colors,
    key.title='', srtCol=45, key.xlab='value')
  return (p)
}

plot_stats <- function(d) {
  d <- d %>% mutate(bin=factor(x))
  p <- ggplot(d, aes(x=bin, y=value_abs)) +
    geom_boxplot(aes(fill=cell_type), outlier.shape=NA) +
    scale_fill_manual(values=colors_$cell_type) +
    xlab('') + ylab('Effect') +
    facet_wrap(~stat, ncol=1, scales='free') +
    theme_pub() +
    theme(legend.position='top') +
    guides(fill=F, size=F, alpha=F)
  return (p)
}

plot_stat <- function(d, stat_, nb_sel=20) {
  d <- d %>% filter(stat == stat_) %>% droplevels
  d$value <- d$value_abs
  f <- d %>% group_by(filt) %>% summarise(value=mean(value, na.rm=T)) %>%
    arrange(desc(value)) %>% head(nb_sel) %>% select(filt) %>% unlist
  d <- d %>% filter(filt %in% f) %>% droplevels %>%
    mutate(filt=factor(filt, levels=f),
      bin=factor(x, levels=sort(unique(x))))
  p <- ggplot(d, aes(x=filt, y=value)) +
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
    facet_wrap(~bin, ncol=1)
  return (p)
}
