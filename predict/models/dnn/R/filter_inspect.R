query_db <- function(db_file, anno=NULL, effect_='abs_lor') {
  con <- src_sqlite(db_file)
  if (is.null(anno)) {
    table <- 'global'
    cond <- NULL
  } else {
    table <- 'annos'
    cond <- sprintf('anno LIKE "%s"', anno)
  }
  h <- sprintf('SELECT * FROM %s', table)
  if (!is.null(cond)) {
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

plot_heat <- function(d, rev_colors=F, Rowv=T, Colv=T, lhei=c(1, 10), del=F) {
  d <- d %>% select(label, target, value) %>%
    spread(target, value) %>% as.data.frame
  rownames(d) <- as.vector(d$label)
  d <- d %>% select(-label) %>% as.matrix(d)

  type <- factor(grepl('2i', colnames(d)), levels=c(T, F), labels=c('2i', 'serum'))
  col_colors <- colors_$cell_type[type]

  if (del) {
    colors <- brewer.pal(11, 'RdBu')
  } else {
    colors <- rev(brewer.pal(9, 'YlOrRd'))
  }
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


to_matrix <- function(d) {
  d <- d %>% mutate(value=mean_) %>% select(label, target, value) %>%
    spread(target, value) %>% as.data.frame
  rownames(d) <- as.vector(d$label)
  d <- d %>% select(-label) %>% as.matrix
  return (d)
}

plot_pca_target <- function(pc, x=1, y=2) {
  t <- data.frame(
    sample=factor(rownames(pc$vec)),
    pcx=pc$vec[,x], pcy=pc$vec[,y]
    ) %>% mutate(
      cell_type=parse_cell_type(sample),
      sample_short=parse_sample_short(sample)
      )
  p <- ggplot(t, aes(x=pcx, y=pcy)) +
    geom_point(aes(color=cell_type)) +
    scale_color_manual(values=colors_$cell_type) +
    geom_text(aes(label=sample_short), vjust=-.4, hjust= .3, size=3) +
    xlab(sprintf('PC%d (%.2f%%)', x, pc$val[x])) +
    ylab(sprintf('PC%d (%.2f%%)', y, pc$val[y])) +
    theme_pub()
  return (p)
}

plot_pca_filt <- function(pc, x=1, y=2) {
  t <- data.frame(
    sample=factor(rownames(pc$vec)),
    pcx=pc$vec[,x], pcy=pc$vec[,y]
    ) %>% mutate(
      sample_short=sub('^(\\d+).+', '\\1', sample)
      )
  p <- ggplot(t, aes(x=pcx, y=pcy)) +
    geom_point() +
    geom_text(aes(label=sample), vjust=-.4, hjust= .3, size=3) +
    xlab(sprintf('PC%d (%.2f%%)', x, pc$val[x])) +
    ylab(sprintf('PC%d (%.2f%%)', y, pc$val[y])) +
    theme_pub()
  return (p)
}

plot_scatter <- function(d) {
  p <- ggplot(d, aes(x=value, y=mean_)) +
    stat_smooth(method=lm, color='black') +
    geom_point(aes(color=cell_type), size=0.9) +
    scale_color_manual(values=colors_$cell_type) +
    facet_grid(label~fact, scale='free') +
    xlab('Factor') + ylab('Importance') +
    theme_pub()
  return (p)
}
