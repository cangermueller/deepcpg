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

parse_cell_type <- function(x) {
  x <- as.vector(x)
  x <- factor(grepl('2i', x), levels=c(T, F), labels=c('2i', 'serum'))
  return (x)
}

char_to_factor <- function(d) {
  for (n in names(d)) {
    if (is.character(d[[n]])) {
      d[[n]] <- factor(d[[n]])
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
