filter_region <- function(d, region) {
  chr <- sub('chr', '', region$chromo)
  d <- d %>% filter(
    chromo == chr,
    pos >= region$start,
    pos <= region$end
  )
  return (d)
}

data_annos <- function(d) {
  d <- d %>% select(name, chromo, start, end) %>%
    gather(pos, x, -c(name, chromo)) %>%
    arrange(name, chromo, pos) %>% tbl_df
  return (d)
}

plot_annos <- function(d) {
  p <- geom_vline(data=d, aes(xintercept=x, linetype=name))
  return (p)
}

data_seqmut <- function(d, region, seqmut_='rnd', wlen_=10) {
  d <- d %>% filter_region(region) %>%
    filter(seqmut == seqmut_, wlen == wlen_) %>%
    droplevels
  return (d)
}

plot_seqmut <- function(d, span=0.05, degree=2, what='lor') {
  p <- ggplot(d, aes_string('x'='pos', 'y'=what)) +
    geom_smooth(aes(group=target, color=cell_type), size=0.5, se=F,
      method='loess', span=span, method.args=list(degree=degree)) +
    theme_pub() +
    ylab('Effect') + xlab('') +
    scale_color_manual(values=colors_$cell_type) +
    scale_x_continuous(labels=comma) +
    theme(axis.title.x=element_blank(), legend.position='top')
  return (p)
}

data_filt_imp <- function(d, region, n=10, excl=NULL) {
  d <- d %>% filter_region(region)
  if (!is.null(excl)) {
    d <- d %>% filter(!(filt %in% excl))
  }
  filts <- d %>% group_by(filt) %>% summarise(value=mean(abs(lor))) %>%
    arrange(desc(value)) %>% head(n) %>% select(filt) %>% unlist
  d <- d %>% filter(filt %in% filts) %>%
    mutate(filt=factor(filt, levels=filts)) %>%
    group_by(chromo, pos, filt) %>%
    summarise(
      lor=mean(lor),
      del=mean(del),
      abs_lor=mean(abs(lor)),
      abs_del=mean(abs(del))
    ) %>% ungroup
  d <- d %>% group_by(filt) %>% mutate(ipos=1:n()) %>% ungroup
  return (d)
}

plot_filt_imp <- function(d, what='lor') {
  p <- ggplot(d, aes(x=ipos, y=filt)) +
    geom_tile(aes_string('fill'=what)) +
    scale_fill_gradient2(low='blue', mid='white', high='red') +
    xlab('') + ylab('Filter') +
    theme(
      panel.border=element_blank(),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.background = element_blank(),
      axis.title.x=element_blank(),
      legend.position='bottom'
      ) +
    guides(fill=F)
  return (p)
}

plot_rates_conf <- function(d, region, targets=NULL, span=0.05) {
  d <- d %>% filter(pos >= region[1], pos <= region[2])
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







plot_bed_tracks <- function(tracks, region) {
  plotTracks(tracks, transcriptAnnotation='symbol',
    from=region$start, to=region$end, margin=50)
}

plot_var_metric <- function(d, span=0.05, degree=2, rates=NULL) {
  cols <- c('chromo', 'pos', 'key', 'metric', 'value')
  d <- d %>%
    gather(key, value, y, z) %>%
    mutate(key=factor(key, levels=c('y', 'z'), labels=c('True', 'Predicted')))
  d <- d[, cols]
  p <- ggplot(d, aes(x=pos, y=value))

  if (!is.null(rates)) {
    r <- rates %>% group_by(chromo, pos) %>%
      summarise(mean=mean(z), var=var(z)) %>% ungroup %>%
      gather(metric, value, mean, var) %>%
      mutate(key='Computed')
    r <- r[, cols]
    p <- p +
      stat_smooth(data=r, aes(color=key), color='grey',
        size=0.7, se=F, method='loess', span=span,
        method.args=list(degree=degree))
  }
  p <- p +
    geom_point(aes(color=key), size=0.4) +
    stat_smooth(aes(color=key),
      size=1, se=F, method='loess', span=span,
      method.args=list(degree=degree)) +
    scale_color_manual(values=colors_$yz) +
    facet_wrap(~metric, ncol=1, scale='free') +
    xlab('') + ylab('') +
    theme_pub() +
    theme(axis.title.x=element_blank(), legend.position='top') +
    scale_x_continuous(labels=comma)
  return (p)
}

plot_rates_metric <- function(d, span=0.05, degree=2) {
  d <- d %>% group_by(chromo, pos, cell_type) %>%
    summarise(var_=var(z)) %>% ungroup
  p <- ggplot(d, aes(x=pos, y=var_)) +
    stat_smooth(aes(color=cell_type),
      size=1, se=F, method='loess', span=span,
      method.args=list(degree=degree)) +
    scale_color_manual(values=colors_$cell_type) +
    guides(color=F) +
    xlab('') + ylab('Variance') +
    theme_pub() +
    theme(axis.title.x=element_blank(), legend.position='top') +
    scale_x_continuous(labels=comma)
  return (p)
}

plot_rates <- function(d, span=0.05, degree=2) {
  p <- ggplot(d, aes(x=pos, y=z)) +
    geom_smooth(aes(group=target, color=target, linetype=cell_type),
      size=0.5, se=F, method='loess', span=span,
      method.args=list(degree=degree)) +
    theme_pub() +
    guides(color=F, linetype=F) +
    ylab('Methylation rate') + xlab('') +
    scale_x_continuous(labels=comma) +
    theme(axis.title.x=element_blank())
  return (p)
}

plot_filt_act <- function(d, center=T, tomtom=NULL, sort_=F) {
  d <- d %>% group_by(filt) %>%
    mutate(
      pos_min=floor(0.5 * (pos + lag(pos))),
      pos_max=ceiling(0.5 * (pos + lead(pos)))
      ) %>% ungroup
  for (n in c('pos_min', 'pos_max')) {
    h <- is.na(d[[n]])
    d[h,][[n]] <- d[h,]$pos
  }
  d <- d %>% mutate(
        x=round(0.5 * (pos_min + pos_max)),
        width=pos_max - pos_min
      )

  if (is.null(tomtom)) {
    d$label <- d$filt
  } else {
    h <- levels(d$filt)
    d <- d %>% left_join(select(tomtom, filt, label=name), by='filt') %>%
      mutate(
        label=sprintf('%d: %s', as.numeric(as.vector(filt)), label),
        label=sub(': NA', '', label),
        filt=factor(filt, levels=h),
        label=factor(label)
        )
  }

  if (sort_) {
    h <- d %>% group_by(filt) %>% summarise(value=mean(value)) %>%
      arrange(value) %>% select(filt) %>% unlist %>% as.vector
    d <- d %>% mutate(filt=factor(filt, levels=h))
  } else {
    d <- d %>% mutate(filt=factor(filt, levels=rev(levels(filt))))
  }
  d$label <- map_factor_order(d$label, d$filt)

  if (center) {
    d <- d %>% group_by(filt) %>%
      mutate(value=(value-mean(value))) %>% ungroup
  }

  p <- ggplot(d) +
    geom_tile(aes(x=x, width=width, y=label, fill=value)) +
    scale_fill_gradient(low='white', high='red') +
    theme_pub() +
    xlab('') + ylab('') +
    theme(
      panel.border=element_blank(),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.background = element_blank(),
      axis.title.x=element_blank(),
      axis.text.y=element_text(size=rel(0.6)),
      legend.position='bottom'
      ) +
    guides(fill=F)
  return (p)
}
