read_group <- function(path, group) {
  d <- list()
  for (k in c('y', 'z')) {
    d[[k]] <- h5read(path, sprintf('%s/%s', group, k))
  }
  d <- as.data.frame(d) %>% tbl_df %>% mutate(name=basename(group))
  return (d)
}

read_pred_target <- function(path, target, annos_regex=NULL) {
  d <- list()
  d[[length(d) + 1]] <- read_group(path, sprintf('/%s/global', target))
  if (!is.null(annos_regex)) {
    annos <- h5ls(path, sprintf('/%s/annos', target))
    annos <- grep(annos_regex, annos, value=T)
    for (anno in annos) {
      d[[length(d) + 1]] <- read_group(path, sprintf('/%s/annos/%s', target, anno))
    }
  }
  d <- do.call(rbind.data.frame, d) %>%
    mutate(name=factor(name)) %>% tbl_df
  return (d)
}

read_pred <- function(path, targets=NULL, ...) {
  if (is.null(targets)) {
    targets <- h5ls(path, '/')
  }
  d <- list()
  for (target in targets) {
    h <- read_pred_target(path, target, ...) %>% mutate(target=target)
    d[[length(d) + 1]] <- h
  }
  d <- do.call(rbind.data.frame, d) %>% mutate(target=factor(target))
  return (d)
}

perf_curve_ <- function(d, x.axis='fpr', y.axis='tpr') {
  p <- prediction(d$z, d$y)
  p <- performance(p, y.axis, x.axis)
  d <- data.frame(x=p@x.values[[1]], y=p@y.values[[1]]) %>% tbl_df
  d <- d[complete.cases(d),]
  return (d)
}

perf_curve <- function(d, x.axis='fpr', y.axis='tpr', cell_type=TRUE) {
  if (cell_type) {
    d <- d %>% group_by(method, cell_type)
  } else {
    d <- d %>% group_by(method)
  }
  d <- d %>%
    do(perf_curve_(., x.axis, y.axis)) %>%
    arrange(x) %>%
    ungroup
  return (d)
}

curve_data <- function(d, nb_sample=NULL) {
  if (!is.null(nb_sample)) {
    if ('cell_type' %in% names(d)) {
      d <- d %>% group_by(method, cell_type)
    } else {
      d <- d %>% group_by(method)
    }
    d <- d %>% sample_n(nb_sample) %>% arrange(x) %>% ungroup
  }
  d <- d %>% filter(y > 0.001)
  return (d)
}

plot_roc <- function(d, ...) {
  d <- curve_data(d, ...)
  p <- ggplot(d, aes(x=x, y=y)) +
    geom_abline(slope=1, linetype='dashed', color='lightgrey') +
    geom_line(aes(color=method), size=1.3) +
    scale_color_manual(values=colors_$model) +
    xlab('False Positive Rate') + ylab('True Positive Rate') +
    theme_pub() +
    guides(color=guide_legend(title='Method')) +
    theme(legend.position='top')
  if ('cell_type' %in% names(d)) {
    p <- p + facet_wrap(~cell_type, scale='free')
  }
  return (p)
}

plot_recall <- function(d, ...) {
  d <- curve_data(d, ...)
  p <- ggplot(d, aes(x=x, y=y)) +
    geom_smooth(aes(color=method), size=1.3, se=F) +
    scale_color_manual(values=colors_$model) +
    xlab('Recall') + ylab('Precision') +
    theme_pub() +
    guides(color=guide_legend(title='Method')) +
    theme(legend.position='top')
  if ('cell_type' %in% names(d)) {
    p <- p + facet_wrap(~cell_type, scale='free')
  }
  return (p)
}
