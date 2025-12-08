# Collaborator Invitation Guide

This guide explains how to invite collaborators to the GenImage-Detector repository without manually asking each person for their username.

## Why Use Invite Links?

Instead of collecting usernames one by one, you can:
- **Share a single link** that multiple people can use
- **Save time** by not asking each collaborator for their username
- **Make it easier** for collaborators to join your project

## Methods to Invite Collaborators

### Method 1: Share the Repository URL (Recommended for Open Collaboration)

Simply share this link:
```
https://github.com/mhmdelbadry1/GenImage-Detector
```

When people visit, they can:
- ⭐ Star the repository to show interest
- 👀 Watch the repository for updates
- 🍴 Fork the repository to contribute via pull requests
- 🐛 Open issues to report bugs or request features

### Method 2: Use the Python Script

Run the included script to generate all useful links:

```bash
python generate_invite_link.py
```

This displays:
- Repository URL
- Issues page URL
- Fork URL
- Settings page (for you to manage access)

### Method 3: GitHub's Built-in Invite Feature (For Direct Collaborators)

For users who need **write access** to the repository:

1. Navigate to: https://github.com/mhmdelbadry1/GenImage-Detector/settings/access
2. Click **"Add people"** or **"Invite a collaborator"**
3. Enter their GitHub username or email address
4. Choose their permission level:
   - **Read**: Can view and clone
   - **Triage**: Can manage issues and pull requests
   - **Write**: Can push to the repository
   - **Maintain**: Can manage the repository without access to sensitive settings
   - **Admin**: Full access to the repository
5. Click **"Add [username] to this repository"**
6. They'll receive an email invitation

### Method 4: Create a "Request Access" Issue Template

Create an issue template that interested collaborators can use:

1. Create `.github/ISSUE_TEMPLATE/request_access.md`:

```markdown
---
name: Request Collaborator Access
about: Request to become a collaborator on this repository
title: '[ACCESS REQUEST] Your Name'
labels: access-request
assignees: mhmdelbadry1
---

## Collaborator Access Request

**GitHub Username:** @your-username

**Why I want to collaborate:**
[Explain your interest in the project]

**Areas I can contribute to:**
- [ ] Code development
- [ ] Documentation
- [ ] Testing
- [ ] Design
- [ ] Other: [specify]

**Background/Experience:**
[Brief description of your relevant experience]
```

Then share: https://github.com/mhmdelbadry1/GenImage-Detector/issues/new/choose

### Method 5: Use GitHub Discussions (If Enabled)

Enable Discussions in your repository settings, then share:
```
https://github.com/mhmdelbadry1/GenImage-Detector/discussions
```

Create a pinned discussion for collaboration requests.

## Best Practices

### For Open Source Projects
1. **Public Repository**: Keep it public so anyone can fork and contribute
2. **Clear CONTRIBUTING.md**: Document how people can contribute
3. **Issue Templates**: Make it easy for people to report bugs or suggest features
4. **Pull Request Template**: Guide contributors on how to submit PRs
5. **Code of Conduct**: Establish community guidelines

### For Private/Controlled Collaboration
1. **Collect Interest First**: Use issues or discussions
2. **Review Profiles**: Check contributors' GitHub profiles
3. **Start with Read Access**: Grant limited permissions initially
4. **Upgrade as Needed**: Increase permissions based on contributions
5. **Use Teams**: For larger projects, organize collaborators into teams

## Quick Command Reference

```bash
# Generate invite links
python generate_invite_link.py

# For a different repository
python generate_invite_link.py <owner> <repo-name>
```

## Troubleshooting

**Q: People can't find the repository**
- Make sure the repository is public, or
- Send them the direct link: https://github.com/mhmdelbadry1/GenImage-Detector

**Q: Invited collaborators didn't receive the email**
- Check they entered the correct email
- Ask them to check spam folder
- They can also check https://github.com/settings/organizations for pending invitations

**Q: I want to revoke access**
- Go to Settings > Collaborators and teams
- Find the user and click "Remove"

## Additional Resources

- [GitHub Docs: Inviting Collaborators](https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-personal-account-on-github/managing-access-to-your-personal-repositories/inviting-collaborators-to-a-personal-repository)
- [GitHub Docs: Permission Levels](https://docs.github.com/en/organizations/managing-user-access-to-your-organizations-repositories/repository-roles-for-an-organization)
- [GitHub Docs: Managing Access](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/managing-teams-and-people-with-access-to-your-repository)

---

**Need help?** Open an issue at: https://github.com/mhmdelbadry1/GenImage-Detector/issues
