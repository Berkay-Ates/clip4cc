# clip4cc

## Getting Started with Local Development

### Prerequisites

Before proceeding, ensure you have installed `uv`, which manages the local development environment. You can find installation instructions in the official `uv` documentation: https://docs.astral.sh/uv/

### Installing Dependencies

This project has several optional dependencies based on the version of PyTorch you wish to use. Choose one of the following options based on your hardware capabilities:

- **`cpu`**: Use this option for PyTorch without CUDA support.
- **`cu118`**: Use this option for PyTorch with CUDA 11.8 support.
- **`cu124`**: Use this option for PyTorch with CUDA 12.4 support.

To set up the full environment, you need to explicitly define one of the options above. Use the following command:

```bash
uv sync --extra XXX
```

Replace `XXX` with one of the options listed above (e.g., `cpu`, `cu118`, or `cu124`).

### Example Command

For example, if you want to set up the environment with CUDA 11.8 support, you would run:

```bash
uv sync --extra cu118
```

## Additional Information

- Ensure that your system meets the requirements for the selected PyTorch version.
- For further details on using `clip4cc`, please refer to the documentation or the examples provided in this repository.

## Contributing

We welcome contributions! Please read our [Contributing Guidelines](CONTRIBUTING.md) for more information on how to get involved.

## License

This project is licensed under the [MIT License](LICENSE).
