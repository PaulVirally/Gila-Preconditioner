Take a look at [example.jl](example.jl) for a simple example of how to
use
[`GilaElectromagnetics.jl`](https://github.com/moleskySean/GilaElectromagnetics.jl)
to generate some data for the models. To make more data, you'll
probably want to use different sources (the vector that the
`LippmannSchwinger` operator acts on) and different media. The
`create_random_medium` function in [`spheres.jl`](spheres.jl) is a
good starting point for you if you want to customize what media you
want to use.
