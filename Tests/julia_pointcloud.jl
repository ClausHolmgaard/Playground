using Plots
# pyplot()
gr(format="png")

N = 1000000

x = randn(N)
y = randn(N)
z = sin.(x.*y)

scatter(x,y,z, markersize=.1, legend=false)
