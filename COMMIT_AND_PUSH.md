# How to Commit & Push NeoRank to GitHub

This guide walks you through committing this complete repository scaffold to GitHub.

## Prerequisites

- Git installed and configured
- GitHub account (github.com/print-EddyMa)
- SSH keys set up OR GitHub personal access token ready

## Step 1: Verify Repository Status

```bash
cd /workspaces/NeoRank

# Check what files are present
ls -la

# Verify git is initialized (should exist)
ls -la .git

# See what's untracked
git status
```

## Step 2: Stage All Files

```bash
git add .
```

## Step 3: Create Initial Commit

```bash
git commit -m "Initial scaffold: Complete NeoRank repository structure

- Created complete directory structure per specification
- Implemented Config, DataPreprocessor, and FeatureExtractor classes
- Added comprehensive documentation (6 docs + FAQ)
- Created test suite framework with sample tests
- Added training and prediction scripts
- Included Jupyter notebook placeholders
- Setup all configuration files (setup.py, requirements.txt, etc.)
- Ready for NetMHCpan integration and model training"
```

## Step 4: Set Remote Repository

### Option A: Push to Existing Repository

If `print-EddyMa/NeoRank` already exists on GitHub:

```bash
# Add origin
git remote add origin https://github.com/print-EddyMa/NeoRank.git

# Or with SSH (if keys configured)
git remote add origin git@github.com:print-EddyMa/NeoRank.git

# Verify
git remote -v
```

### Option B: Create New Repository First

Go to https://github.com/new and:

1. Set Repository name: `NeoRank`
2. Set Description: "Accessible Neoantigen Immunogenicity Prediction"
3. Choose: Public
4. Do NOT initialize with README
5. Click "Create repository"

Then follow Option A above.

## Step 5: Push to GitHub

### Using HTTPS (with token):

```bash
git branch -M main
git push -u origin main
```

You'll be prompted for:
- Username: print-EddyMa
- Password: [use Personal Access Token, not GitHub password]

### Using SSH:

```bash
git branch -M main
git push -u origin main
```

(Requires SSH keys configured on GitHub)

### Using GitHub CLI:

If you have `gh` installed:

```bash
gh repo create print-EddyMa/NeoRank \
  --public \
  --source=. \
  --remote=origin \
  --push
```

## Step 6: Verify on GitHub

Visit: https://github.com/print-EddyMa/NeoRank

You should see:
- ✓ All files in the repo
- ✓ Full directory structure
- ✓ README.md displayed
- ✓ LICENSE shown

## Troubleshooting

### "fatal: remote origin already exists"

```bash
# Remove existing remote
git remote remove origin

# Then re-add
git remote add origin https://github.com/print-EddyMa/NeoRank.git
```

### "fatal: 'origin' does not appear to be a 'git' repository"

Make sure you're in the `/workspaces/NeoRank` directory:

```bash
cd /workspaces/NeoRank
pwd  # should show /workspaces/NeoRank
```

### Authentication Failed

#### For HTTPS:
- Use a Personal Access Token (Settings → Developer Settings → Personal Access Tokens)
- Do NOT use your GitHub password

#### For SSH:
- Check SSH key is added: `ssh -T git@github.com`
- Should see: "Hi print-EddyMa! You've successfully authenticated"

### Branch name conflicts

```bash
# If 'main' doesn't exist, create it
git branch -M main

# Then push
git push -u origin main
```

## Next Steps After Push

Once successfully pushed:

1. **Verify on GitHub**
   - Visit your repo
   - Check all files are there
   - Test git clone works

2. **Set up GitHub Pages (Optional)**
   - Settings → Pages
   - Choose `main` branch
   - GitHub will host docs automatically

3. **Add GitHub Secrets (For CI/CD)**
   - Settings → Secrets and Variables → Actions
   - Add any required tokens

4. **Set up Branch Protection (Optional)**
   - Settings → Branches
   - Add rule for `main`
   - Require PR reviews before merge

5. **Create Issues for Next Steps**
   - File issues for model training implementation
   - Track feature requests
   - Organize work with GitHub Projects

## Example Full Workflow

```bash
cd /workspaces/NeoRank

# 1. Check status
git status

# 2. Add all files
git add .

# 3. Create commit
git commit -m "Initial scaffold: Complete NeoRank repository"

# 4. Set remote
git remote add origin https://github.com/print-EddyMa/NeoRank.git

# 5. Push to GitHub
git branch -M main
git push -u origin main

# 6. Verify
git log --oneline
git remote -v
```

## Additional Resources

- Git Docs: https://git-scm.com/doc
- GitHub Docs: https://docs.github.com
- SSH Setup: https://docs.github.com/en/authentication/connecting-to-github-with-ssh
- Personal Access Token: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token

---

**Status**: ✅ Repository ready to push
**Next Action**: Run commit and push commands above
**Questions?**: See docs/ directory for more info
