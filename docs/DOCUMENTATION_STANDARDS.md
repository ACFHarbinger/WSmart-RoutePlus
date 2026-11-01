# WSmart-Route Documentation Standards

**Version**: 1.0
**Created**: February 2026
**Purpose**: Unified style guide for all module documentation

---

## Document Structure Template

### 1. Header Block (Required)

```markdown
# [Module Name]

**Module**: `path/to/module`
**Purpose**: [Single sentence describing module purpose]
**Version**: 3.0
**Last Updated**: February 2026
```

### 2. Table of Contents (Required)

- Numbered list (1, 2, 3...)
- Maximum 3 levels deep (1.1.1)
- Use descriptive section names
- Include all major sections

### 3. Overview Section (Required)

**Contents:**
- Brief introduction (2-3 paragraphs)
- Key Features (bullet list)
- Design Principles (bullet list, if applicable)
- Import Strategy (code example)

### 4. Module Organization (Required for multi-file modules)

**Contents:**
- Directory structure (code block with tree format)
- Key components table
- Dependency graph (optional, if complex)
- File statistics (optional)

### 5. Main Content Sections

**Guidelines:**
- Clear, descriptive section headers
- Subsections with consistent formatting
- Code examples with syntax highlighting
- Tables for structured reference data
- Mathematical notation using LaTeX (inline: `$...$`, block: `$$...$$`)

### 6. Integration Examples (Required)

**Contents:**
- Practical usage scenarios
- Complete, runnable code examples
- Expected outputs
- Common patterns

### 7. Best Practices (Required)

**Structure:**
```markdown
### ✅ Good Practices

[Example with explanation]

### ❌ Anti-Patterns

[Counter-example with explanation]

### Common Pitfalls

[Issue description with solution]
```

### 8. Quick Reference (Required)

**Contents:**
- Common imports
- Command summary (if applicable)
- File locations table
- Cross-references to related documentation

### 9. Footer (Required)

```markdown
---

**Last Updated**: [Month Year]
**Maintainer**: WSmart Route+ Development Team
**Status**: ✅ Active
```

---

## Style Guidelines

### Tone

- **Professional and technical** (avoid dramatic language)
- **Precise and concise** (clear technical descriptions)
- **Instructional** (explain "how" and "why")

### Formatting

#### Headers

- Use `#` for title, `##` for major sections, `###` for subsections
- Use sentence case (capitalize first word and proper nouns)
- Keep headers concise (max 60 characters)

#### Code Blocks

````markdown
```python
# Always include language identifier
# Add brief comment explaining code purpose
def example_function():
    return "value"
```
````

#### Tables

```markdown
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Value    | Value    | Value    |
```

- Always include header row
- Align columns consistently
- Use descriptive column names

#### Lists

- **Bullet lists**: Use `-` for consistency
- **Numbered lists**: Use `1.` format
- **Nested lists**: Indent with 2 spaces

#### Emphasis

- **Bold** (`**text**`): For definitions, emphasis
- *Italic* (`*text*`): For terminology, variables
- `Code` (`` `text` ``): For code, file paths, commands

### Mathematical Notation

- Use LaTeX format for math:
  - Inline: `$x + y = z$`
  - Block display: `$$\sum_{i=1}^{n} x_i$$`
- Define symbols on first use
- Use consistent notation throughout documentation

### Cross-References

```markdown
See [Section Name](#section-anchor) for details.
See [Other Module](OTHER_MODULE.md) for related information.
```

- Use descriptive link text
- Verify links work
- Prefer relative paths for internal links

### Emoji Usage

**Minimal and purposeful only:**
- ✅ for "good" examples
- ❌ for "bad" examples
- ⚠️ for warnings
- 💡 for tips
- 📝 for notes

**Avoid:**
- Decorative emojis
- Emojis in section headers (except Best Practices)
- Excessive emoji usage

---

## Content Guidelines

### Code Examples

1. **Always include:**
   - Context (what problem it solves)
   - Complete, runnable code
   - Expected output or behavior
   - Edge cases (if relevant)

2. **Code quality:**
   - Follow project coding standards
   - Include type hints
   - Add explanatory comments
   - Keep examples concise but complete

### API Documentation

**For each function/class:**
- Purpose (brief description)
- Parameters table
- Return value description
- Usage example
- Notes on edge cases

### Performance Considerations

Document when relevant:
- Time complexity
- Space complexity
- GPU memory requirements
- Scalability limits

### Security Considerations

Document when applicable:
- Input validation requirements
- Security implications
- Best practices
- Common vulnerabilities

---

## Module-Specific Guidelines

### CLI Module

- Document all commands and subcommands
- Include usage examples with output
- Show both simple and complex use cases

### Configuration Module

- Show hierarchy visually
- Document all configuration options
- Provide complete examples
- Explain defaults and valid ranges

### Implementation Modules (Policies, Models, etc.)

- Document design patterns used
- Explain architectural decisions
- Include performance characteristics
- Provide integration examples

### Utility Modules

- Group by functionality
- Clear API reference
- Show common use cases
- Document error handling

---

## Quality Checklist

Before finalizing documentation:

- [ ] All required sections present
- [ ] Table of contents accurate and complete
- [ ] Code examples tested and working
- [ ] Links verified
- [ ] Mathematical notation correct
- [ ] Tables properly formatted
- [ ] Consistent tone throughout
- [ ] No spelling/grammar errors
- [ ] Cross-references accurate
- [ ] Quick reference section complete

---

## Examples

### Good Example: Section Structure

```markdown
## Configuration Loading

The configuration system uses Hydra for hierarchical composition.

### Basic Usage

```python
from logic.src.configs import Config

# Load default configuration
config = Config()

# Override specific values
config = Config(
    env=EnvConfig(num_loc=100),
    device="cuda"
)
```

### Advanced Patterns

...
```

### Bad Example: Inconsistent Formatting

```markdown
## configuration loading!!! 🎉

here's how to load configs:

config = Config() # just do this

Now you can use it anywhere!
```

---

**Maintainer**: WSmart Route+ Development Team
**Last Updated**: February 2026
