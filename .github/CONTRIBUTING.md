# Contributing to XRTailor

All kinds of contributions are welcome, including but not limited to the following.

- Fixes (typo, bugs)
- New features and components

## Workflow

1. Fork and pull the latest xrtailor
1. Checkout a new branch with a meaningful name (do not use master branch for PRs)
1. Commit your changes
1. Create a PR

```{note}
- If you plan to add some new features that involve large changes, it is encouraged to open an issue for discussion first.
- If you are the author of some papers and would like to include your method to xrtailor, please contact us. We will much appreciate your contribution.
```

## Code style

We follow the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html). We use the following tools for linting and formatting:
- [clang-tidy](https://clang.llvm.org/extra/clang-tidy/): linter

- [clang-format](https://clang.llvm.org/docs/ClangFormat.html): formatter

Style configurations of clang-tidy can be found in [.clang-tidy](../.clang-tidy) and clang-format in [.clang-format](../clang-format).

> Before creating a PR, make sure your code passes clang-tidy checks and is formatted with clang-format.