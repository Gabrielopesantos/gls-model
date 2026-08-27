{ pkgs, lib, config, ... }:

{
  # Python 3.12: torch ships cp312 wheels, and it is the widest-supported
  # version across lm-eval/transformers/wandb.
  languages.python = {
    enable = true;
    package = pkgs.python312;
    venv.enable = true;

    uv = {
      enable = true;
      sync = {
        enable = true;
        # Only the dev group by default. The eval and track groups
        # are installed on demand with `uv sync --group`.
        groups = [ "dev" ];
      };
    };

    # Native libraries the PyPI wheels dlopen at runtime. The CUDA runtime
    # itself is not here: torch's bundled nvidia-* wheels carry it via RPATH.
    libraries = [
      pkgs.stdenv.cc.cc.lib # libstdc++
      pkgs.zlib
    ];
  };

  # libcuda.so belongs to the kernel driver, not to any package, so it has to
  # come off the driver link. Without this, torch.cuda.is_available() is False
  # inside the shell even though nvidia-smi works outside it.
  env.LD_LIBRARY_PATH = lib.mkAfter "${pkgs.addDriverRunpath.driverLink}/lib";

  packages = [
    # Large unfree download, and most nsight runs happen on
    # the rented GPU box rather than here.
    # pkgs.cudaPackages.nsight_systems
  ];

  scripts.gpu-check.exec = ''
    python -m gls.env
  '';

  scripts.check.exec = ''
    pytest -q "$@"
  '';

  scripts.fmt.exec = ''
    ruff format . && ruff check --fix .
  '';

  git-hooks.hooks =
    let
      ruff = "${config.devenv.state}/venv/bin/ruff";
    in
    {
      ruff-lint = {
        enable = true;
        name = "ruff check";
        entry = "${ruff} check --fix";
        types = [ "python" ];
        pass_filenames = true;
      };

      ruff-fmt = {
        enable = true;
        name = "ruff format";
        entry = "${ruff} format";
        types = [ "python" ];
        pass_filenames = true;
      };
    };

  enterShell = ''
    if [ -f "$DEVENV_ROOT/.env" ]; then
      set -a
      . "$DEVENV_ROOT/.env"
      set +a
    fi

    echo "gls-model"
  '';

  enterTest = ''
    pytest -q
  '';
}
