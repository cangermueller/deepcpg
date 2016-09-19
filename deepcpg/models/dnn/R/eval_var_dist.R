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

plot_dist_match <- function(d, metric_='mean') {
  d <- d %>% filter(metric == metric_)
  d <- d %>% gather(key, value, y, z)
  p <- ggplot(d, aes(x=anno, y=value)) +
    geom_boxplot(aes(fill=key)) +
    scale_fill_manual(values=colors_$yz) +
    theme_pub() +
    theme(
      legend.position='top'
    ) +
    xlab('') + ylab(metric) +
    facet_wrap(~cell_type, ncol=1) +
    theme(axis.text.x=element_text(angle=30, hjust=1))
  return (p)
}

plot_dist_type <- function(d, metric_='mean') {
  d <- d %>% filter(metric == metric_)
  d <- d %>% gather(key, value, z, y)
  p <- ggplot(d, aes(x=anno, y=value)) +
    geom_boxplot(aes(fill=cell_type)) +
    scale_fill_manual(values=colors_$cell_type) +
    theme_pub() +
    theme(
      axis.text.x=element_text(angle=30, hjust=1),
      legend.position='top'
    ) +
    xlab('') + ylab(metric) +
    facet_wrap(~key, ncol=1)
  return (p)
}
