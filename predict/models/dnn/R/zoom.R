filter_region <- function(d, region) {
  chr <- sub('chr', '', region$chromo)
  d <- d %>% filter(
    chromo == chr,
    pos >= region$start,
    pos <= region$end
  )
  return (d)
}

data_annos <- function(a) {
  d <- do.call(rbind.data.frame, a) %>% mutate(name=factor(names(a))) %>%
    gather(pos, x, -c(name, chromo)) %>%
    arrange(name, chromo, pos) %>% tbl_df
  return (d)
}

plot_annos <- function(d) {
  p <- geom_vline(data=d, aes(xintercept=x, linetype=name))
  return (p)
}

plot_var <- function(d, span=0.05, degree=2) {
  d1 <- d %>% group_by(chromo, pos, cell_type) %>%
    summarise(var_=var(z)) %>% ungroup
  d2 <- d %>% group_by(chromo, pos) %>%
    summarise(var_=var(z)) %>% ungroup %>% mutate(cell_type='all')
  d <- rbind.data.frame(d1, d2) %>% tbl_df
  p <- ggplot(d, aes(x=pos, y=var_)) +
    stat_smooth(aes(linetype=cell_type),
      size=1, se=F, method='loess', span=span,
      method.args=list(degree=degree)) +
    xlab('') + ylab('Variance') +
    theme_pub() +
    scale_x_continuous(labels=comma) +
    theme(axis.title.x=element_blank(), legend.position='top')
  return (p)
}

plot_met <- function(d, span=0.05, degree=2) {
  p <- ggplot(d, aes(x=pos, y=z)) +
    geom_smooth(aes(color=target, linetype=cell_type), size=1, se=F,
      method='loess', span=span, method.args=list(degree=degree)) +
    theme_pub() +
    guides(color=F, linetype=F) +
    ylab('Methylation rate') + xlab('') +
    scale_x_continuous(labels=comma) +
    theme(axis.title.x=element_blank())
  return (p)
}

data_seqmut <- function(d, region, seqmut_='rnd', wlen_=10) {
  d <- d %>% filter_region(region) %>%
    filter(seqmut == seqmut_, wlen == wlen_) %>%
    mutate(value=lor)
  return (d)
}

plot_seqmut <- function(d, span=0.05, degree=2) {
  p <- ggplot(d, aes(x=pos, y=value)) +
    geom_smooth(aes(color=target, linetype=cell_type), size=0.5, se=F,
      method='loess', span=span, method.args=list(degree=degree)) +
    theme_pub() +
    guides(color=F, linetype=F) +
    ylab('Effect') + xlab('') +
    scale_x_continuous(labels=comma) +
    theme(axis.title.x=element_blank())
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
      legend.position='bottom'
      ) +
    theme(axis.title.x=element_blank())
  return (p)
}

data_filt_act <- function(d, region, n=10, excl=NULL) {
  d <- d %>% filter_region(region)
  if (!is.null(excl)) {
    d <- d %>% filter(!(filt %in% excl))
  }
  filts <- d %>% group_by(filt) %>% summarise(value=mean(abs(value))) %>%
    arrange(desc(value)) %>% head(n) %>% select(filt) %>% unlist
  d <- d %>% filter(filt %in% filts) %>%
    mutate(filt=factor(filt, levels=filts)) %>% droplevels
  d <- d %>% group_by(filt) %>% mutate(ipos=1:n()) %>% ungroup
  return (d)
}

plot_filt_act <- function(d) {
  p <- ggplot(d, aes(x=ipos, y=filt)) +
    geom_tile(aes(fill=value)) +
    scale_fill_gradient(low='white', high='red') +
    xlab('') + ylab('Filter') +
    theme(
      panel.border=element_blank(),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.background = element_blank(),
      legend.position='bottom'
      ) +
    theme(axis.title.x=element_blank())
  return (p)
}

plot_met_conf <- function(d, region, targets=NULL, span=0.05) {
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






# Functions with global dependencies

get_plots <- function(region, annos=NULL, span=0.1, nb_filt=20) {
  d <- dat$pred %>% filter_region(region)
  p <- list()
  if (!is.null(annos)) {
    p$annos <- plot_annos(annos)
  }
  p$var <- plot_var(d, span=span) + p$annos
  p$met <- plot_met(d, span=span) + p$annos
  d <- dat$seqmut %>% data_seqmut(
    region=region, seqmut=opts$seqmut, wlen=opts$seqmut_wlen)
  p$seqmut <- plot_seqmut(d) + p$annos
  d <- dat$filt_imp %>% data_filt_imp(region, n=nb_filt, excl=opts$excl)
  p$filt_imp <- plot_filt_imp(d) + p$annos
  d <- data_filt_act(dat$filt_act, region, n=nb_filt, excl=opts$excl)
  p$filt_act <- plot_filt_act(d)
  return (p)
}

plot_grid <- function(p) {
  grid.arrange(p$var, p$met, p$seqmut, p$filt_imp, p$filt_act, ncol=1)
}

get_tracks <- function(region) {
  itrack <- IdeogramTrack(genome=region$genome, chromo=region$chromo)
  atracks <- beds_to_atracks(dat$bed, region)
  if ('Active_enhancers' %in% names(atracks)) {
    atracks$Active_enhancers@name <- 'AE'
  }
  bm <- useMart('ensembl', dataset='mmusculus_gene_ensembl')
  fm <- Gviz:::.getBMFeatureMap()
  grtrack <- BiomartGeneRegionTrack(
    genome=region$genome, chromo=region$chromo, start=region$start, end=region$end,
    name='gene', biomart=bm, featureMap=fm)
  tracks <- c(itrack, atracks, grtrack)
}

plot_tracks <- function(tracks, region) {
  plotTracks(tracks, transcriptAnnotation='symbol',
    from=region$start, to=region$end, margin=50)
}

plot_region <- function(region) {
  plots <- get_plots(region)
  tracks <- get_tracks(region)
  plot_tracks(tracks, region)
  plot_grid(plots)
}
