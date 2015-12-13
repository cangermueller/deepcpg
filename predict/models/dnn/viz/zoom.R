theme_pub <- function() {
  p <- theme(
    axis.text=element_text(size=rel(1.0), color='black'),
    axis.title=element_text(size=rel(1.5)),
    axis.title.y=element_text(vjust=1.0),
    axis.title.x=element_text(vjust=-0.5),
    legend.position='top',
    legend.text=element_text(size=rel(1.0)),
    legend.title=element_text(size=rel(1.0)),
    legend.key=element_rect(fill='transparent'),
    panel.border=element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    axis.line = element_line(colour="black", size=1),
    axis.ticks.length = unit(.3, 'cm'),
    axis.ticks.margin = unit(.3, 'cm')
    )
  return (p)
}

data_annos <- function(a) {
  d <- list()
  for (n in names(a)) {
    an <- a[[n]]
    d[[n]]$name <- n
    d[[n]]$start <- an[1]
    d[[n]]$end <- an[2]
  }
  d <- do.call(rbind.data.frame, d) %>% mutate(name=factor(name)) %>%
    gather(pos, x, -name) %>% tbl_df
  return (d)
}

plot_annos <- function(d) {
  p <- geom_vline(data=d, aes(xintercept=x, linetype=name))
  return (p)
}

plot_var <- function(region, span=0.05) {
  d <- dat$zm %>% filter(pos >= region[1], pos <= region[2])
  d <- d %>% group_by(chromo, pos) %>% summarise(var=var(zm))
  p <- ggplot(d, aes(x=pos, y=var)) +
    geom_smooth(linewidth=0.3, se=F, method='loess', degree=2, span=span) +
    xlab('') + ylab('Variance') +
    theme_pub() +
    scale_x_continuous(labels=comma)
    theme(axis.title.x=element_blank())
  return (p)
}

plot_met <- function(region, span=0.05) {
  d <- dat$zm %>% filter(pos >= region[1], pos <= region[2])
  p <- ggplot(d, aes(x=pos, y=zm)) +
    geom_smooth(aes(color=target, linetype=cell_type), linewidth=0.3, se=F, method='loess', degree=2, span=span) +
    theme_pub() +
    guides(color=F, linetype=F) +
    ylab('Methylation rate') + xlab('') +
    scale_x_continuous(labels=comma) +
    theme(axis.title.x=element_blank())
  return (p)
}

plot_eff_target <- function(region, name=NULL, label=NULL, span=0.05) {
  d <- dat$eff_target %>%
    filter(pos >= region[1], pos <= region[2]) %>%
    select(-c(z, z_mut, zd_del))
  d <- d %>% gather(stat, value, -c(chromo, pos, target, cell_type))
  if (!is.null(name)) {
    d <- d %>% filter(stat == name)
  }
  p <- ggplot(d, aes(x=pos, y=value)) +
    geom_smooth(aes(color=target, linetype=cell_type), linewidth=0.3, se=F, method='loess', degree=2, span=span) +
    guides(color=F, linetype=F) +
    theme_pub() +
    scale_x_continuous(labels=comma) +
    theme(axis.title.x=element_blank())
  if (is.null(name)) {
    facet_wrap(~stat, scale='free', ncol=1)
  }
  return (p)
}

plot_eff <- function(region, span=0.05) {
  d <- dat$eff %>%
    filter(pos >= region[1], pos <= region[2]) %>%
    select(-c(z, z_mut, zd_del))
  d <- d %>% gather(stat, value, -c(chromo, pos))
  p <- ggplot(d, aes(x=pos, y=value)) +
    geom_smooth(linewidth=0.3, se=F, method='loess', degree=2, span=span) +
    facet_wrap(~stat, scale='free', ncol=1) +
    theme_pub() +
    scale_x_continuous(labels=comma) +
    theme(axis.title.x=element_blank())
  return (p)
}

plot_filter_heat <- function(d) {
  d <- d %>% mutate(filter=factor(filter, levels=rev(levels(filter))))
  p <- ggplot(d, aes(x=x, y=filter)) +
    geom_tile(aes(fill=value)) +
    scale_fill_gradient2(low='blue', mid='white', high='red') +
    xlab('') + ylab('Filter') +
    theme(
      panel.border=element_blank(),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.background = element_blank(),
      legend.position='bottom'
      ) +
    scale_x_continuous(aes(labels=pos)) +
    theme(axis.title.x=element_blank())

  return (p)
}

plot_region <- function(region, pa=NULL, span=0.1, delta=30) {
  pv <- plot_var(region, span=span) + pa
  pm <- plot_met(region, span=span) + pa
  pe <- plot_eff_target(region, 'zd_ora', span=span) + ylab('Effect size') + pa
  d <- prepro_filters(dat$sx, delta=delta)
  d <- d %>% filter(pos >= region[1], pos <= region[2])
  pf <- plot_filter_heat(d)
  grid.arrange(pv, pm, pe, pf, ncol=1)
}

plot_met_conf <- function(region, targets=NULL, span=0.05) {
  d <- dat$zm %>% filter(pos >= region[1], pos <= region[2])
  if (!is.null(targets)) {
    d <- d %>% filter(target %in% targets)
  }
  p <- ggplot(d, aes(x=pos, y=zm)) +
    geom_ribbon(aes(ymin=zm-zsd, ymax=zm+zsd, fill=target), alpha=0.2) +
    # geom_smooth(aes(color=target, linetype=cell_type),
    # method='loess', degree=2, linewidth=0.3, span=0.05, se=F) +
    geom_line(aes(color=target, linetype=cell_type), linewidth=0.3) +
    geom_point(aes(y=y, color=target), size=1, alpha=0.5) +
    xlab('') + ylab('Methylation rate') +
    theme_pub() +
    guides(color=F, fill=F, linetype=F) +
    theme(axis.title.x=element_blank()) +
    scale_x_continuous(labels=comma) +
    scale_color_brewer(palette='Dark2') +
    scale_fill_brewer(palette='Dark2')
  return (p)
}
