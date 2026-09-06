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
        # dev + track (wandb) + infra (by default)
        groups = [ "dev" "track" "infra" ];
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

  # Triton locates libcuda by shelling out to a hard-coded /sbin/ldconfig, which
  # does not exist on NixOS. Point it at the driver link (same path as above) so
  # any Triton path - torch.compile, a hand-written kernel, an HF model whose
  # kernels dispatch through Triton - skips that probe.
  env.TRITON_LIBCUDA_PATH = "${pkgs.addDriverRunpath.driverLink}/lib";

  packages = [
    pkgs.nvitop
    pkgs.graphviz # `gls model summary --graph` (torchview) shells out to `dot`
    pkgs.rclone

    # Large unfree download, and most nsight runs happen on
    # the rented GPU box rather than here.
    # pkgs.cudaPackages.nsight_systems
  ];

  scripts.gpu-check.exec = ''
    gls env
  '';

  scripts.check.exec = ''
    pyright && pytest -q "$@"
  '';

  # Thin wrappers over the one `gls` console script, kept for muscle memory.
  scripts.tokenizer.exec = ''
    gls tokenizer "$@"
  '';

  scripts.fmt.exec = ''
    ruff format . && ruff check --fix .
  '';

  git-hooks.hooks =
    let
      ruff = "${config.devenv.state}/venv/bin/ruff";
      pyright = "${config.devenv.state}/venv/bin/pyright";
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

      pyright = {
        enable = true;
        name = "pyright";
        entry = "${pyright}";
        types = [ "python" ];
        pass_filenames = false;
      };
    };

  enterShell = ''
    # Object storage defaults to Backblaze B2; .env overrides either to switch
    # rclone remotes. Mirrored in infra/vast/onstart.sh's profile.d.
    export GLS_REMOTE="''${GLS_REMOTE:-b2}"
    if [ -f "$DEVENV_ROOT/.env" ]; then
      set -a
      . "$DEVENV_ROOT/.env"
      set +a
    fi
    export GLS_BUCKET="''${GLS_BUCKET:-''${GLS_B2_BUCKET:-}}"

    echo "gls-model"
  '';

  enterTest = ''
    pytest -q
  '';
}
