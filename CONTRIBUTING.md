# Contributing to PINNs Project

We welcome contributions to the Physics-Informed Neural Networks for Salt Detection project! This document provides guidelines for contributing to the project.

## 🤝 Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:
- Be respectful and inclusive
- Use welcoming and inclusive language
- Be collaborative and constructive
- Focus on what is best for the community

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Git
- Familiarity with TensorFlow/Keras
- Understanding of deep learning concepts

### Development Setup

1. **Fork the repository**
   ```bash
   git clone https://github.com/Jebin-05/PINNs.git
   cd PINNs
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If available
   ```

4. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 🎯 How to Contribute

### Types of Contributions

1. **Bug Reports** 🐛
   - Report bugs using GitHub Issues
   - Include detailed reproduction steps
   - Provide system information and error logs

2. **Feature Requests** ✨
   - Suggest new features via GitHub Issues
   - Explain the use case and benefits
   - Provide implementation suggestions if possible

3. **Code Contributions** 💻
   - Bug fixes
   - New features
   - Performance improvements
   - Documentation updates

4. **Documentation** 📚
   - Improve README files
   - Add code comments
   - Create tutorials or examples
   - Update API documentation

### Pull Request Process

1. **Before Starting**
   - Check existing issues and PRs
   - Discuss major changes in an issue first
   - Ensure your idea aligns with project goals

2. **Development**
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed
   - Ensure all tests pass

3. **Submission**
   - Create a descriptive PR title
   - Provide detailed description of changes
   - Reference related issues
   - Request review from maintainers

## 🔧 Development Guidelines

### Code Style

**Python Style Guide**
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write descriptive variable and function names
- Keep functions focused and small

**Example:**
```python
def preprocess_image(
    image_path: str, 
    target_size: Tuple[int, int] = (128, 128)
) -> np.ndarray:
    """
    Preprocess seismic image for model input.
    
    Args:
        image_path: Path to the input image
        target_size: Target dimensions for resizing
        
    Returns:
        Preprocessed image array
    """
    # Implementation here
    pass
```

### Documentation Style

- Use clear, concise language
- Include code examples
- Add docstrings for all functions and classes
- Update README for significant changes

### Testing Guidelines

- Write unit tests for new functions
- Include integration tests for major features
- Test edge cases and error conditions
- Maintain high test coverage

### Commit Message Format

Use conventional commit format:
```
type(scope): description

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(model): add ResNet-based classifier

- Implement ResNet architecture for salt classification
- Add training script and configuration
- Include performance benchmarks

fix(ui): resolve image upload validation error

docs(readme): update installation instructions
```

## 🧪 Testing

### Running Tests
```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_utils.py

# Run with coverage
python -m pytest --cov=src tests/
```

### Test Structure
```
tests/
├── unit/
│   ├── test_utils.py
│   ├── test_models.py
│   └── test_preprocessing.py
├── integration/
│   ├── test_training_pipeline.py
│   └── test_prediction_pipeline.py
└── fixtures/
    ├── sample_images/
    └── test_data.py
```

## 📝 Documentation

### Code Documentation
- Add docstrings to all public functions
- Include parameter types and descriptions
- Provide usage examples
- Document return values

### Project Documentation
- Update README for new features
- Add tutorials for complex workflows
- Create API documentation
- Include performance benchmarks

## 🎯 Priority Areas

We're particularly interested in contributions in these areas:

### High Priority
- **Performance Optimization**: GPU utilization, memory efficiency
- **Model Improvements**: New architectures, loss functions
- **Testing**: Comprehensive test coverage
- **Documentation**: User guides, API docs

### Medium Priority
- **Web Interface**: UI/UX improvements
- **Data Pipeline**: Augmentation techniques
- **Monitoring**: Logging and metrics
- **Deployment**: Docker, cloud deployment

### Low Priority
- **Visualizations**: New plotting functions
- **Utilities**: Helper functions
- **Examples**: Tutorial notebooks
- **Benchmarks**: Performance comparisons

## 🐛 Reporting Issues

### Bug Reports
When reporting bugs, please include:

1. **Environment Information**
   - OS and version
   - Python version
   - TensorFlow version
   - GPU information (if applicable)

2. **Steps to Reproduce**
   - Minimal code example
   - Input data description
   - Expected vs actual behavior

3. **Error Information**
   - Full error traceback
   - Log files (if applicable)
   - Screenshots (for UI issues)

### Feature Requests
For feature requests, please provide:

1. **Problem Description**
   - What problem does this solve?
   - Who would benefit from this feature?

2. **Proposed Solution**
   - Detailed description of the feature
   - Alternative solutions considered
   - Implementation suggestions

3. **Additional Context**
   - Related issues or discussions
   - External references
   - Mockups or examples

## 🏅 Recognition

Contributors will be recognized in:
- README acknowledgments
- CONTRIBUTORS.md file
- Release notes
- GitHub contributors graph

## 📞 Getting Help

If you need help with contributing:

1. **Check Documentation**
   - README files
   - Code comments
   - Existing issues/PRs

2. **Ask Questions**
   - Create a GitHub issue with `question` label
   - Join our discussions
   - Contact maintainers

3. **Start Small**
   - Fix typos or improve documentation
   - Add tests for existing code
   - Implement small features first

## 📋 Checklist

Before submitting a PR, ensure:

- [ ] Code follows project style guidelines
- [ ] Tests are added and passing
- [ ] Documentation is updated
- [ ] Commit messages follow convention
- [ ] PR description is clear and detailed
- [ ] Related issues are referenced
- [ ] No merge conflicts exist

## 🎉 Thank You!

Thank you for contributing to the PINNs project! Your contributions help advance the field of physics-informed deep learning and geological imaging.

---

For questions about contributing, please open an issue or contact the maintainers.
