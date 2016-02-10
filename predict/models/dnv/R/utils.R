format_target <- function(d) {
    d <- d %>%
      separate(target, c('cell_type', 'wlen', 'type'), by='_', remove=F) %>%
      mutate(
        cell_type=factor(cell_type, levels=c('2i', 'ser'), labels=c('2i', 'serum')),
        wlen=as.integer(sub('w', '', wlen))
      )
    return (d)
}
