theme_pub <- function() {
  p <- theme(
    axis.text=element_text(size=rel(1.0), color='black'),
    axis.title=element_text(size=rel(1.5)),
    axis.title.y=element_text(vjust=1.0),
    axis.title.x=element_text(vjust=-0.5),
    legend.position='right',
    legend.text=element_text(size=rel(1.0)),
    legend.title=element_text(size=rel(1.0)),
    legend.key=element_rect(fill='transparent'),
    panel.border=element_blank(),
    # panel.grid.major = element_blank(),
    # panel.grid.minor = element_blank(),
    panel.grid.major=element_line(colour="grey60", size=0.1, linetype='solid'),
    panel.grid.minor=element_line(colour="grey60", size=0.1, linetype='dotted'),
    panel.background = element_blank(),
    axis.line = element_line(colour="black", size=1),
    axis.ticks.length = unit(.3, 'cm'),
    axis.ticks.margin = unit(.3, 'cm')
    )
  return (p)
}

colors_ <- list()
colors_$cell_type <- c(
  '2i'='#377eb8',
  'serum'='#e41a1c'
  )
colors_$yz <- c('y'='#1b9e77', 'z'='#d95f02')
colors_$yz <- c(
  'y'='#e6ab02',
  'True'='#e6ab02',
  'z'='#66a61e',
  'Predicted'='#66a61e'
  )

colors_$model <- c(
    'DeepCpG'='#1b9e77',
    'RF'='#d95f02',
    'WinAvg'='#fb9a99',
    'DeepCpG Seq'='#66a61e',
    'RF Seq'='#7570b3')

linetypes <- list()
linetypes$cell_type <- c(
  '2i'='dashed',
  'serum'='solid'
  )

annos_ <- list()
annos_$perf_short <- c(
  'loc_loc_TSSs', 'loc_Exons', 'loc_H3K27me3', 'loc_dnase1', 'loc_CGI',
  'loc_p300', 'loc_gene_body', 'loc_CGI_shore', 'loc_Introns', 'loc_Intergenic',
  'loc_prom_2k05k_ncgi', 'loc_H3K27ac', 'loc_Active_enhancers',
  'loc_mESC_enhancers', 'loc_H3K4me1', 'loc_LMRs')
annos_$perf_long <- c('loc_prom_2k05k', 'loc_TSSs', 'loc_Exons',
  'loc_prom_2k05k_cgi', 'psu_dnase1', 'licr_H3k27ac', 'loc_H3K27me3',
  'licr_H3k09ac', 'loc_Wu_Tet1', 'licr_P300', 'uw_dnase1', 'licr_H3k04me3',
  'loc_CGI', 'licr_Ctcf', 'loc_CGI_shelf', 'loc_p300', 'loc_gene_body',
  'loc_Tet2', 'loc_CGI_shore', 'licr_H3k04me1', 'global', 'loc_Introns',
  'licr_Pol2', 'loc_Intergenic', 'loc_Oct4_2i', 'licr_H3k36me3',
  'loc_prom_2k05k_ncgi', 'loc_H3K27ac', 'loc_Active_enhancers',
  'loc_H3K4me1_Tet1', 'loc_mESC_enhancers', 'loc_H3K4me1', 'loc_IAP',
  'loc_LMRs')

gopts <- list()
gopts$lo_better <- c('mse', 'rmse', 'loss')

parse_cell_type <- function(x) {
  x <- as.vector(x)
  x <- factor(grepl('2i', x), levels=c(T, F), labels=c('2i', 'serum'))
  return (x)
}

parse_sample_short <- function(x) {
  return (sub('.+(RSC.+)', '\\1', x))
}

parse_var_target <- function(d) {
    d <- d %>%
      separate(target, c('cell_type', 'wlen', 'metric'), by='_', remove=F) %>%
      mutate(
        cell_type=factor(cell_type,
          levels=c('2i', 'ser'),
          labels=c('2i', 'serum')),
        wlen=as.integer(sub('w', '', wlen))
      ) %>% droplevels
    return (d)
}

format_annos <- function(annos) {
  annos <- sub('(loc_|licr_|chipseq_|stan_)', '', annos)
  return (annos)
}

char_to_factor <- function(d) {
  for (n in names(d)) {
    if (is.character(d[[n]])) {
      d[[n]] <- factor(d[[n]])
    }
  }
  return (d)
}

factor_to_char <- function(d) {
  for (n in names(d)) {
    if (is.factor(d[[n]])) {
      d[[n]] <- as.character(d[[n]])
    }
  }
  return (d)
}

rename_values <- function(x, how) {
  for (nn in names(how)) {
    no <- how[nn]
    h <- x == no
    if (any(h)) {
      x[h] <- nn
    }
  }
  return (x)
}

move_cols <- function(d, cols) {
  h <- setdiff(colnames(d), cols)
  d <- d[,c(cols, h)]
  return (d)
}

move_front <- function(x, what) {
  h <- c(what, setdiff(x, what))
  h <- intersect(h, x)
  return (h)
}

read_report_values <- function(filename, samples=NULL, n=NULL) {
  h <- 'cut -f 13-'
  if (!is.null(n)) {
    h <- sprintf('head -n %d %s | %s', n, filename, h)
  } else {
    h <- sprintf('%s %s', h, filename)
  }

  h <- read.table(pipe(h), head=T, sep='\t')
  if (!is.null(samples)) {
    h <- subset(h, select=intersect(colnames(h), samples))
  }
  h <- h %>% tbl_df
  return (h)
}

read_report_meta <- function(filename, n=NULL) {
  sel <- 'cut -f 2-5,7,8,12'
  if (!is.null(n)) {
    cmd <- sprintf('head -n %d %s | %s', n, filename, sel)
  } else {
    cmd <- sprintf('%s %s', sel, filename)
  }
  h <- read.table(pipe(cmd), head=T, sep='\t')
  names(h) <- tolower(names(h))
  h <- h %>% rename(chromo=chromosome)
  h <- h %>% tbl_df
  return (h)
}

impute <- function(d) {
  means <- colMeans(d, na.rm=T)
  if (any(is.na(means))) {
    stop('Insufficient data for mean imputation!')
  }
  for (i in 1:length(means)) {
    d[is.na(d[,i]), i] <- means[i]
  }
  return (d)
}

pca <- function(d, center=T, scale=F) {
  # columns are samples
  d <- scale(d, center=center, scale=scale)
  d <- t(d)
  s <- svd(d)
  vec <- as.data.frame(s$u)
  colnames(vec) <- sapply(1:ncol(vec), function(x) sprintf('PC%d', x))
  rownames(vec) <- rownames(d)
  val <- s$d**2
  val <- val / sum(val)
  return (list(vec=vec, val=val))
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

h5ls <- function(path, group='/') {
  if (group[1] != '/') {
    group <- paste0('/', group)
  }
  h <- sprintf('h5ls %s%s', path, group)
  h <- system(h, intern=T)
  h <- sapply(strsplit(h, '\\s+'), function(x) x[1])
  return (h)
}

add_global <- function(to, global, name=c('anno'='global')) {
  global[[names(name)[1]]] <- name[1]
  to <- rbind.data.frame(global, to) %>% mutate(anno=factor(anno))
  return (to)
}

grep_all <- function(a, b, value=T) {
  h <- NULL
  for (aa in a) {
    hh <- grep(aa, b, value=F)
    if (is.null(h)) {
      h <- hh
    } else {
      h <- union(h, hh)
    }
  }
  h <- sort(h)
  if (value) {
    h <- b[h]
  }
  return (h)
}
