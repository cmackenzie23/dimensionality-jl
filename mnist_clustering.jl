using MLDatasets, Plots, UMAP

# MNIST Cluster Analysis with UMAP

train = MNIST(:train)

processed = reshape(train.features, (28*28, 60000))

#heatmap(reshape(processed[5,:], (28,28))', color=:grays, aspect_ratio=1, yflip=true, show=true)

embedding = umap(processed, n_neighbors=15, min_dist=0.1)

c_palette = palette(:tab10)
scatter()
for i in 0:9
	scatter!(embedding[1, train.targets.==i], embedding[2, train.targets.==i], marker = :circle, markerstrokewidth = 0, color = c_palette[i+1], markersize=0.5, label="$(i)")
end
gui()