#!/usr/bin/env Rscript


library(argparse)
library(ggplot2)
library(dplyr)
library(tidyr)
library(rhdf5)
library(grid)
library(seqLogo)

p <- ArgumentParser(description='Visualize filters')
p$add_argument(
  'in_file',
  help='Input file')
p$add_argument(
  '--filter_name',
  default='s_c1',
  help='Filter name')
p$add_argument(
  '-o', '--out_logos',
  default='./filters.pdf',
  help='Name of output logos file')
p$add_argument(
  '--out_heat',
  help='Name of output heatmap file')
p$add_argument(
  '--verbose',
  action='store_true',
  help='More detailed log messages',
  default=F)

args <- commandArgs(TRUE)
if (length(args) > 0) {
  opts <- p$parse_args(args)
} else {
  opts <- list()
  opts$in_file <- '../filters.h5'
  opts$filter_name <- 's_c1'
  opts$out_logos <- './logos.pdf'
  opts$out_heat <- './heat.pdf'
  opts$verbose <- 0
}
if (opts$verbose == 1) {
  print(opts)
}

dat <- list()


log <- function(x) {
  cat(x, fill=T)
}

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

filter_frame <- function(f) {
  f <- f %>% as.data.frame
  names(f) <- 1:ncol(f)
  f$char <- c('A', 'G', 'T', 'C')
  f <- f %>% gather(pos, value, -char) %>% tbl_df
  return (f)
}

read_filters <- function(path, group='/s_c1') {
  d <- h5read(path, group)
  fs <- list()
  for (i in 1:dim(d)[4]) {
    f <- filter_frame(t(d[,,,i]))
    f$filter <- i
    fs[[length(fs) + 1]] <- f
  }
  f <- do.call(rbind.data.frame, fs) %>% tbl_df
  f <- f %>% rename(act=value)
  return (f)
}

plot_heat_filters <- function(d, what='act', negative=T) {
  p <- ggplot(d, aes(x=pos, y=char)) +
    geom_tile(aes_string('fill'=what)) +
    facet_wrap(~filter, ncol=3, scale='free') +
    xlab('Position') + ylab('Nucleotide') +
    theme(
      panel.border=element_blank(),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.background = element_blank(),
      legend.position='right'
      )
  if (negative) {
    p <- p + scale_fill_gradient2(low='blue', mid='white', high='red')
  } else {
    p <- p + scale_fill_gradient(low='white', high='red')
  }
  return (p)
}


log('Read data')
dat$fil <- read_filters(opts$in_file, group=paste0('/', opts$filter_name))
dat$fil <- dat$fil %>%
  group_by(filter) %>% mutate(
    act_m=act-mean(act),
    act_ms=(act-mean(act))/sd(act)
  ) %>% ungroup %>%
  group_by(filter, pos) %>% mutate(
    act_p1=exp(act_m)/sum(exp(act_m)),
    act_p2=exp(act_ms)/sum(exp(act_ms))
    ) %>% ungroup %>%
  mutate(
    filter=factor(filter, levels=unique(sort(as.numeric(filter)))),
    pos=factor(pos, levels=unique(sort(as.numeric(pos))))
    )
dat$nb_filter <- length(levels(dat$fil$filter))
dat$filter_len <- length(levels(dat$fil$pos))

if (!is.null(opts$out_heat)) {
  log('Plot heat map')
  p <- plot_heat_filters(dat$fil, 'act_ms')
  ggsave(p, file=opts$out_heat,
    width=dat$filter_len * 0.8 * 3,
    height=ceiling(dat$nb_filter / 3) * 2
    )
}

get_pmw <- function(fil, what='act_p2') {
  d <- dat$fil %>% filter(filter==fil) %>%
    select_('pos', 'char', 'value'=what) %>%
    spread(pos, value)
  h <- d$char
  d <- d %>% select(-char) %>% as.matrix
  rownames(d) <- h
  return (d)
}

if (!is.null(opts$out_logos)) {
  cat('Plot sequence logos')
  pdf(opts$out_logos,
    width=dat$filter_len * 1.0,
    height=5
    )
  for (i in 1:dat$nb_filter) {
    d <- get_pmw(i, 'act_p2')
    seqLogo(d)
  }
  dev.off()
}

log('Done!')
