library(dplyr)

index_to_matrix<- function (column){
  matrix <- matrix(0, ncol= 780, nrow= 4987)
  for (l in 1:4987){
    seq <- as.integer(unlist(strsplit(column[l], "")))
    for (i in 1:20){
      matrix[l,i] <- seq[i]
    }
    k <- 21
    for (i in 1:19){
        for (j in (i+1):20){
           t<-seq[[i]]
           m<-seq[[j]]
           fr<-k + t * 2 + m
           matrix[l,fr]<-1
           k <- k + 4
        }
    }
  }
  matrix
}



data<-read.csv("df.csv",  colClasses = c("character", rep("numeric", 30)))
colnames(data)[1] <- "seq"
data<- data %>% remove_rownames %>% column_to_rownames(var="seq")
#print(as.numeric(unlist(strsplit(row.names(data)[1], "")))[[1]])
mat<- index_to_matrix(row.names(data))

indices <- list()
for (i in 1:20){
  indices <- append(indices, as.character(i))
}
for (i in 1:19){
  for (j in (i+1):20){
      indices <- append(indices, paste(as.character(i),as.character(j),'00', sep="_"))
      indices <- append(indices, paste(as.character(i),as.character(j),'01', sep="_"))
      indices <- append(indices, paste(as.character(i),as.character(j),'10', sep="_"))
      indices <- append(indices, paste(as.character(i),as.character(j),'11', sep="_"))
  }
}

df<- data.frame(matrix(0,nrow = 780, ncol = length(colnames(data))))
colnames(df)<-colnames(data)
row.names(df)<-indices
for (i in seq_along(colnames(df))){
  df["1", i] <- i
}
sum_of_obs <- colSums(data)
seq_weights <- apply(data, 2, function(x) x/sum_of_obs)
count_freq_for_p <- function(column) {
  p <- column[1]
  ind <- colnames(df)[p]
  k <- seq_weights[,ind]
  l <- t(mat) %*% k
  column<- unlist(as.list(l))
  column
}

df <- df%>%mutate(across(colnames(df),count_freq_for_p))
dfl <- df%>%mutate_all(log1p)

