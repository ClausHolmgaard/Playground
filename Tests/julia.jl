using Plots
using ORCA
#pyplot()
#plotlyjs()
plotly()
#gr()

a_dict = Dict("test" => "a test",
              "one" => 1,
              "two" => 2)

x = 1:10; y = rand(10); # These are the plotting data
#gcf()
p = plot(x, y, reuse=false, size=(900, 550));
#display(p)
gui()
println("Plot done")
