query_db <- function(db_file, anno=NULL, effect_='abs_lor') {
  con <- src_sqlite(db_file)
  if (is.null(anno)) {
    table <- 'global'
    cond <- NULL
  } else {
    table <- 'annos'
    cond <- sprintf('anno = "%s"', anno)
  }
  h <- sprintf('SELECT * FROM %s', table)
  if (!is.null(anno)) {
    h <- sprintf('%s WHERE %s', h, cond)
  }
  d <- tbl(con, sql(h))
  d <- d %>% collect %>% select(-c(path, id))
  d <- d %>% filter(effect == effect_) %>% select(-effect)
  d <- d %>% mutate(cell_type=parse_cell_type(target))
  d <- d %>% mutate(fun=paste0(fun, '_')) %>% spread(fun, value)
  d <- d %>% rename(filt=seqmut) %>%
    mutate(filt=factor(as.numeric(gsub('^z_f', '', filt))))
  d <- d %>% group_by(target) %>% mutate(rank_=rank(-abs(mean_))) %>%
    ungroup
  d <- d %>% char_to_factor %>% droplevels %>% tbl_df %>%
    move_cols(c('filt', 'target', 'cell_type'))
  return (d)
}

plot_global <- function(d) {
  d <- d %>% mutate(label=factor(label, levels=rev(levels(label))))
  p <- ggplot(d, aes(x=label, y=mean_)) +
    geom_boxplot(aes(fill=cell_type), outlier.shape=NA) +
    geom_point(aes(fill=cell_type), size=0.8,
      position=position_jitterdodge(jitter.width=0.5, jitter.height=0, dodge.width=0.8)) +
    scale_fill_manual(values=colors_$cell_type) +
    guides(color=F) +
    xlab('') + ylab('') +
    theme_pub() +
    coord_flip()
  return (p)
}

plot_heat <- function(d, rev_colors=F, Rowv=T, Colv=T, lhei=c(1, 10)) {
  d <- d %>% select(label, target, value) %>%
    spread(target, value) %>% as.data.frame
  rownames(d) <- as.vector(d$label)
  d <- d %>% select(-label) %>% as.matrix(d)

  type <- factor(grepl('2i', colnames(d)), levels=c(T, F), labels=c('2i', 'serum'))
  col_colors <- colors_$cell_type[type]

  colors <- brewer.pal(9, 'Spectral')
  if (!rev_colors) {
    colors <- rev(colors)
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
