read_group <- function(path, group) {
  d <- list()
  for (k in c('y', 'z')) {
    d[[k]] <- h5read(path, sprintf('%s/%s', group, k))
  }
  d <- as.data.frame(d) %>% tbl_df %>% mutate(name=basename(group))
  return (d)
}

read_pred <- function(path, target, annos_regex=NULL) {
  d <- list()
  d[[length(d) + 1]] <- read_group(path, sprintf('/%s/global', target))
  annos <- h5ls(path, sprintf('/%s/annos', target))
  if (!is.null(annos_regex)) {
    annos <- grep(annos_regex, annos, value=T)
  }
  for (anno in annos) {
    d[[length(d) + 1]] <- read_group(path, sprintf('/%s/annos/%s', target, anno))
  }
  d <- do.call(rbind.data.frame, d) %>%
    mutate(name=factor(name)) %>% tbl_df
  return (d)
}

plot_dist_type <- function(d) {
  d <- d %>% gather(key, value, z, y)
  p <- ggplot(d, aes(x=anno, y=value)) +
    geom_boxplot(aes(fill=cell_type)) +
    scale_fill_manual(values=colors_$cell_type) +
    theme_pub() +
    theme(
      axis.text.x=element_text(angle=30, hjust=1),
      legend.position='top'
    ) +
    ylim(0, 0.15) +
    xlab('') + ylab('Variance') +
    facet_wrap(~key, ncol=1)
  return (p)
}

plot_dist_match <- function(d, vert=F) {
  d <- d %>% gather(key, value, y, z)
  if (vert) {
    levels(d$anno) <- rev(levels(d$anno))
  }
  p <- ggplot(d, aes(x=anno, y=value)) +
    geom_boxplot(aes(fill=key)) +
    scale_fill_brewer(palette='Set4') +
    theme_pub() +
    theme(
      legend.position='top'
    ) +
    ylim(0, 0.15) +
    xlab('') + ylab('Variance') +
    coord_flip()
  if (vert) {
  } else {
    p <- p +
      theme(
        axis.text.x=element_text(angle=30, hjust=1)
        )
  }
  return (p)
}

format_target <- function(d) {
    d <- d %>%
      separate(target, c('cell_type', 'wlen', 'type'), by='_', remove=F) %>%
      mutate(
        cell_type=factor(cell_type, levels=c('2i', 'ser'), labels=c('2i', 'serum')),
        wlen=as.integer(sub('w', '', wlen))
      )
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
    gather(fun, value, 1:9)
  d <- d %>% format_target
  d <- d %>% char_to_factor %>% droplevels %>% tbl_df %>%
    move_cols(c('model', 'target', 'cell_type', 'wlen'))
  return (d)
}

order_annos <- function(d, by='mse') {
  h <- d %>% filter(fun==by) %>%
    group_by(anno) %>%
    summarise(value=mean(value)) %>%
    arrange(value)
  return (as.vector(h$anno))
}

plot_yz <- function(d, wlen_=3000, funs=c('y', 'z', 'ymed', 'zmed')) {
  d <- d %>% filter(wlen==wlen_)
  h <- order_annos(d)
  d <- d %>% filter(fun %in% funs) %>%
    mutate(fun=factor(fun, levels=funs))
  d$anno <- factor(d$anno, levels=rev(h))
  p <- ggplot(d, aes(x=anno, y=value)) +
    geom_bar(aes(fill=cell_type), stat='identity', position='dodge') +
    scale_fill_manual(values=colors_$cell_type) +
    theme_pub() +
    theme(
      axis.text.x=element_text(angle=30, hjust=1),
      legend.position='top'
    ) +
    xlab('') + ylab('Variance') +
    facet_wrap(~fun, ncol=1)
  return (p)
}

plot_scores <- function(d, wlen_=3000, funs=c('mse', 'mad', 'cor')) {
  d <- d %>% filter(wlen==wlen_, fun %in% funs) %>%
    mutate(fun=factor(fun, levels=funs))
  d$anno <- factor(d$anno, levels=rev(order_annos(d)))
  p <- ggplot(d, aes(x=anno, y=value)) +
    geom_bar(aes(fill=cell_type), stat='identity', position='dodge') +
    scale_fill_manual(values=colors_$cell_type) +
    facet_wrap(~fun, ncol=1, scale='free') +
    theme_pub() +
    theme(
      axis.text.x=element_text(angle=30, hjust=1),
      legend.position='top'
    ) +
    xlab('') + ylab('Variance')
  return (p)
}
