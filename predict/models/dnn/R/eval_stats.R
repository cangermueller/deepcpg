parse_wlen <- function(s) {
  w <- grepl('_', s)
  h <- sapply(s[!w], function(x) sprintf('w0_%s', x))
  s[!w] <- h
  return (s)
}

query_db <- function(db_file, table='global', cond=NULL, uniq=T) {
  con <- src_sqlite(db_file)
  h <- sprintf('SELECT * FROM %s', table)
  if (!is.null(cond)) {
    h <- sprintf('%s WHERE %s', h, cond)
  }
  d <- tbl(con, sql(h)) %>% collect
  if (uniq) {
    # Remove identical records
    h <- lapply(names(d), as.symbol)
    d <- d %>% group_by_(.dots=h) %>% slice(1) %>% ungroup
  }
  d <- d %>% select(-path, -id)
  d <- d %>% gather(fun, value, c(mean, median, min, max, n))
  d <- d %>% mutate(stat=parse_wlen(stat)) %>%
    separate(stat, c('wlen', 'stat'), sep='_') %>%
    mutate(wlen=as.numeric(sub('w', '', wlen)))
  d <- d %>% char_to_factor %>% droplevels %>% tbl_df
  return (d)
}

read_data <- function(db_files) {
  ds <- list()
  for (g in names(db_files)) {
    db_file <- db_files[g]
    d <- query_db(db_file, 'annos')
    h <- query_db(db_file, 'global') %>% mutate(anno='global')
    d <- rbind.data.frame(d, h) %>% mutate(group=g)
    ds[[length(ds) + 1]] <- d
  }
  ds <- do.call(rbind.data.frame, ds) %>%
    mutate(group=factor(group, levels=names(db_files)))
  return (ds)
}

plot_counts <- function(d) {
  d <- d %>% filter(anno != 'global')
  h <- d %>% group_by(anno) %>% summarise(n=mean(n)) %>%
    arrange(desc(n)) %>% select(anno) %>% unlist
  d <- d %>% mutate(anno=factor(anno, levels=rev(h)))
  p <- ggplot(d, aes(x=anno, y=n)) +
    geom_bar(aes(fill=group), stat='identity', position='dodge') +
    scale_fill_manual(values=colors_$groups) +
    xlab('') + ylab('') +
    theme_pub() + theme(legend.position='top') +
    coord_flip()
  return (p)
}

plot_annos <- function(d, wlen_, win_stats, fun_='mean') {
  d <- d %>%
    filter(wlen == wlen_, stat %in% win_stats, fun==fun_) %>%
    droplevels %>%
    mutate(stat=factor(stat, levels=win_stats))
  h <- d %>% filter(stat == win_stats[1]) %>% group_by(anno) %>%
    summarise(value=mean(value)) %>% arrange(value) %>% select(anno) %>% unlist
  d <- d %>% mutate(anno=factor(anno, levels=rev(h)))
  p <- ggplot(d, aes(x=anno, y=value)) +
    geom_bar(aes(fill=group), stat='identity', position='dodge') +
    scale_fill_manual(values=colors_$groups) +
    xlab('') + ylab('') +
    theme_pub() +
    theme(
      legend.position='top',
      axis.text.x=element_text(angle=30, hjust=1)
      ) +
    facet_wrap(~stat, scale='free', ncol=1)
  return (p)
}

plot_wlen <- function(d, anno_='global', fun_='mean') {
  d <- d %>% filter(anno == anno_, fun==fun_, wlen > 0) %>% droplevels
  p <- ggplot(d, aes(x=wlen, y=value)) +
    geom_line(aes(color=group)) +
    geom_point(aes(color=group)) +
    scale_color_manual(values=colors_$groups) +
    xlab('Window length') + ylab('Value') +
    theme_pub() +
    theme(legend.position='top') +
    facet_wrap(~stat, ncol=3, scale='free')
  return (p)
}
