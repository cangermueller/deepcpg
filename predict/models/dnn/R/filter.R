query_db_ <- function(db_file, table='global', cond=NULL) {
  con <- src_sqlite(db_file)
  h <- sprintf('SELECT * FROM %s', table)
  if (!is.null(cond)) {
    h <- sprintf('%s WHERE %s', h, cond)
  }
  d <- tbl(con, sql(h))
  d <- d %>% collect %>% select(-path, -id)
  if ('filt' %in% names(d)) {
    d$filt <- factor(d$filt)
  }
  d <- d %>% char_to_factor %>% droplevels %>% tbl_df
  return (d)
}

query_db <- function(db_path, global=T, annos=T) {
  d <- list()
  if (global) {
    d$global <- query_db_(db_path, 'global') %>% mutate(anno='global')
  }
  if (annos) {
    d$annos <- query_db_(db_path, 'annos')
  }
  n <- NULL
  for (i in 1:length(d)) {
    if (is.null(n)) {
      n <- names(d[[i]])
    } else {
      n <- intersect(n, names(d[[i]]))
    }
  }
  for (i in 1:length(d)) {
    d[[i]] <- d[[i]][,n]
  }
  d <- do.call(rbind.data.frame, d) %>%
    mutate(anno=factor(anno)) %>%
    droplevels %>% tbl_df
  return (d)
}

filt_to_frame <- function(f) {
  f <- f %>% as.data.frame
  names(f) <- ncol(f):1
  f$char <- c('A', 'G', 'T', 'C')
  f$char <- factor(f$char, levels=f$char)
  f <- f %>% gather(pos, value, -char) %>% tbl_df
  return (f)
}

read_filt <- function(path, group='/filter/weights') {
  d <- h5read(path, group)
  fs <- list()
  for (i in 1:dim(d)[4]) {
    f <- filt_to_frame(t(d[1,,,i]))
    f$filt <- i - 1
    fs[[length(fs) + 1]] <- f
  }
  f <- do.call(rbind.data.frame, fs) %>% tbl_df
  f <- f %>% rename(act=value) %>%
    group_by(filt) %>% mutate(
      act_m=act-mean(act),
      act_ms=(act-mean(act))/sd(act)
    ) %>% ungroup %>%
    group_by(filt, pos) %>% mutate(
      act_p1=exp(act_m)/sum(exp(act_m)),
      act_p2=exp(act_ms)/sum(exp(act_ms))
      ) %>% ungroup %>%
    mutate(
      filt=factor(filt, levels=unique(sort(as.numeric(filt)))),
      pos=factor(pos, levels=unique(sort(as.numeric(pos))))
      )
  return (f)
}

plot_filt_heat <- function(d, what='act', negative=T) {
  p <- ggplot(d, aes(x=pos, y=char)) +
    geom_tile(aes_string('fill'=what)) +
    facet_wrap(~filt, ncol=3, scale='free') +
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

filt_pwm <- function(d, filt_, what='act_p2') {
  d <- d %>% filter(filt==filt_) %>%
    select_('pos', 'char', 'value'=what) %>%
    mutate(char=factor(char, levels=c('A', 'C', 'G', 'T'))) %>%
    spread(pos, value)
  h <- d$char
  d <- d %>% select(-char) %>% as.matrix
  rownames(d) <- h
  return (d)
}

filt_motif <- function(d, filt) {
  d <- filt_pwm(d, filt)
  return (paste0(rownames(d)[apply(d, 2, which.max)], collapse=''))
}





filt_distance <- function(d) {
  eps <- 1e-6
  s <- d %>% mutate(
      y=pmin(1 - eps, pmax(eps, z)),
      y0=pmin(1 - eps, pmax(eps, z0))
    ) %>% summarise(
      z_del=mean(z - z0),
      z_abs=mean(abs(z - z0)),
      z_lor=mean(log2(y / (1 - y)) - log2(y0 / (1 - y0))),
      z_alor=mean(abs(log2(y / (1 - y)) - log2(y0 / (1 - y0))))
    )
  return (s)
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

read_filt_imp_target <- function(path, target, filt, chromos=NULL) {
  if (is.null(chromos)) {
    chromos <- h5ls(path, target)
  }
  ds <- list()
  for (chromo in chromos) {
    h <- c('pos', 'z', filt)
    p <- file.path(target, chromo)
    d <- lapply(h, function(x) as.vector(h5read(path, file.path(p, x))))
    names(d) <- c('pos', 'z', 'z0')
    d <- as.data.frame(d)
    d$chromo <- chromo
    ds[[length(ds) + 1]] <- d
  }
  ds <- do.call(rbind.data.frame, ds) %>% tbl_df
  return (ds)
}

read_filt_imp <- function(path, targets=NULL, filts=NULL) {
  if (is.null(targets)) {
    targets <- h5ls(path)
  }
  chromos <- h5ls(path, targets[1])
  if (is.null(filts)) {
    filts <- h5ls(path, file.path(targets[1], chromos[1]))
    filts <- grep('z_f.*', filts, value=T)
  }

  ds <- list()
  for (target in targets) {
    for (filt in filts) {
      d <- read_filt_imp_target(path, target, filt, chromos)
      d <- filt_distance(d)
      d <- d %>% mutate(
        target=target,
        nfilt=filt,
        filt=as.numeric(gsub('z_f', '', filt)))
      ds[[length(ds) + 1]] <- d
    }
  }
  ds <- do.call(rbind.data.frame, ds) %>% tbl_df %>%
    mutate(filt=factor(filt))
  return (ds)
}

