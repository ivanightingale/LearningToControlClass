using Documenter
using Pluto

include("definitions.jl")

repo_dir = dirname(@__DIR__)
build_dir = joinpath(repo_dir, "docs", "build")

plutos = [
    joinpath(repo_dir, "class01", "background_materials", "basics_math.jl"),
]

symlink(joinpath(repo_dir, "class01", "class01.md"),
    joinpath(repo_dir, "docs", "src", "class01.md")
) 

makedocs(
    sitename = "LearningToControlClass",
    format = Documenter.HTML(;
        assets = ["assets/wider.css", "assets/redlinks.css"],
        mathengine = Documenter.MathJax3(Dict(
            :tex => Dict(
                "macros" => make_macros_dict("docs/src/assets/definitions.tex"),
                "inlineMath" => [["\$","\$"], ["\\(","\\)"]],
                "tags" => "ams",
            ),
        )),
    ),
    pages  = [
        "Home"   => "index.md",
        "Class 1" => "class01.md",
    ],
)

s = Pluto.ServerSession();
for pluto in plutos
    nb = Pluto.SessionActions.open(s, pluto; run_async=false)
    html_contents = Pluto.generate_html(nb; binder_url_js="undefined")
    filename = replace(pluto, repo_dir => build_dir)
    html_path = replace(filename, r"\.jl$" => ".html")
    mkpath(dirname(html_path))
    open(html_path, "w") do f
        write(f, html_contents)
    end
end

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo="github.com/LearningToOptimize/LearningToControlClass.git",
    push_preview=true,
)
