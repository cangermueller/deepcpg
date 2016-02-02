query_db <- function(db_file, table='global', cond=NULL) {
  con <- src_sqlite(db_file)
  h <- sprintf('SELECT * FROM %s', table)
  if (!is.null(cond)) {
    h <- sprintf('%s WHERE %s', h, cond)
  }
  d <- tbl(con, sql(h))
  d <- d %>% collect %>% select(-path, -id)
  d <- d %>% mutate(cell_type=parse_cell_type(target),
    value_del=r, value_abs=abs(r))
  d <- d %>% char_to_factor %>% droplevels %>% tbl_df
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
  p <- ggplot(d, aes(x=filt, y=value_abs)) +
    geom_boxplot(aes(fill=cell_type), outlier.shape=NA) +
    geom_point(aes(fill=cell_type), size=0.8,
      position=position_jitterdodge(jitter.width=0.5, jitter.height=0, dodge.width=0.8)) +
    scale_fill_manual(values=colors_$cell_type) +
    guides(color=F) +
    xlab('') + ylab('Effect') +
    theme_pub() +
    theme(legend.position='top')
  return (p)
}

plot_heat <- function(d, rev_colors=F, Rowv=T, Colv=T, lhei=c(1, 10), del=F) {
  f <- ifelse(del, 'value_del', 'value_abs')
  d$value <- d[[f]]
  d <- d %>% select(filt, target, value) %>% spread(target, value) %>%
    as.data.frame
  rownames(d) <- d$filt
  d <- d %>% select(-filt) %>% as.matrix

  type <- factor(grepl('2i', colnames(d)), levels=c(T, F), labels=c('2i', 'serum'))
  col_colors <- colors_$cell_type[type]

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
    margins=c(8, 8), lhei=lhei,
    key.title='', srtCol=45, key.xlab='value', ColSideColors=col_colors)
  return (p)
}

plot_annos <- function(d) {
  p <- ggplot(d, aes(x=filt, y=value_abs)) +
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
    facet_wrap(~anno, ncol=1)
  return (p)
}

plot_annos_distinct <- function(dat) {
  for (anno_ in astats$anno) {
    d <- dat %>% filter(anno == anno_) %>% droplevels
    f <- d %>% group_by(filt) %>% summarise(value_abs=mean(value_abs)) %>%
      arrange(desc(value_abs)) %>% head(opts$nb_sel) %>% select(filt) %>% unlist
    d <- d %>% filter(filt %in% f) %>% droplevels %>%
      mutate(filt=factor(filt, levels=f))
    print(plot_annos(d))
  }
}

plot_annos_mean <- function(d) {
  p <- ggplot(d, aes(x=anno, y=value_abs)) +
    geom_boxplot(aes(fill=cell_type), outlier.size=0) +
    scale_fill_manual(values=colors_$cell_type) +
    theme_pub() +
    theme(
      axis.text.x=element_text(angle=30, hjust=1),
      legend.position='top'
      ) +
    xlab('') + ylab('Effect')
  return (p)
}

plot_annos_heat <- function(d, rev_colors=F, Rowv=T, Colv=T, lhei=c(1, 10), del=F) {
  f <- ifelse(del, 'value_del', 'value_abs')
  d$value <- d[[f]]
  d <- d %>% group_by(anno, filt) %>% summarise(value=mean(value)) %>%
    spread(anno, value) %>% as.data.frame
  rownames(d) <- d$filt
  d <- d %>% select(-filt) %>% as.matrix

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
    margins=c(8, 8), lhei=lhei,
    key.title='', srtCol=45, key.xlab='value')
  return (p)
}

plot_stats <- function(d) {
  d <- d %>% group_by(stat, x, cell_type, filt) %>%
    summarise(value=mean(value_abs)) %>% ungroup
  d <- d %>% mutate(bin=factor(x))
  p <- ggplot(d, aes(x=bin, y=value)) +
    geom_boxplot(aes(fill=cell_type), outlier.shape=NA) +
    scale_fill_manual(values=colors_$cell_type) +
    # geom_text(aes(label=filt, size=exp(r)),
    #   position=position_jitter(width=0.3)) +
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
