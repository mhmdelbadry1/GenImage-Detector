#!/usr/bin/env python3
"""
GitHub Repository Invite Link Generator

This script helps you generate shareable links for inviting collaborators
to your GitHub repository without needing to ask for their usernames individually.
"""

import sys


def generate_invite_links(repo_owner, repo_name):
    """
    Generate various shareable links for a GitHub repository.
    
    Args:
        repo_owner: The GitHub username or organization name
        repo_name: The repository name
    
    Returns:
        dict: Dictionary containing different types of links
    """
    base_url = f"https://github.com/{repo_owner}/{repo_name}"
    
    links = {
        'repository': base_url,
        'issues': f"{base_url}/issues",
        'discussions': f"{base_url}/discussions",
        'fork': f"{base_url}/fork",
        'settings_access': f"{base_url}/settings/access",
    }
    
    return links


def print_links(links):
    """Print the generated links in a user-friendly format."""
    print("\n" + "="*70)
    print("GitHub Repository Invite Links")
    print("="*70)
    print("\n📌 Share these links with potential collaborators:\n")
    
    print(f"🔗 Repository URL:")
    print(f"   {links['repository']}")
    print(f"   Share this main link with collaborators to view the repository\n")
    
    print(f"🐛 Issues Page:")
    print(f"   {links['issues']}")
    print(f"   Collaborators can report issues here\n")
    
    print(f"💬 Discussions:")
    print(f"   {links['discussions']}")
    print(f"   For community discussions (if enabled)\n")
    
    print(f"🍴 Fork Repository:")
    print(f"   {links['fork']}")
    print(f"   Users can fork and contribute via pull requests\n")
    
    print("="*70)
    print("\n⚙️  To add collaborators with write access:")
    print(f"   1. Go to: {links['settings_access']}")
    print(f"   2. Click 'Add people' or 'Invite teams or people'")
    print(f"   3. Enter their GitHub username or email")
    print(f"   4. Select their permission level")
    print(f"   5. Send invitation\n")
    
    print("💡 Alternative: Enable GitHub Discussions or use Issues")
    print("   to let people express interest, then you can invite them!\n")
    print("="*70 + "\n")


def main():
    """Main function to generate and display invite links."""
    # Default repository information
    repo_owner = "mhmdelbadry1"
    repo_name = "GenImage-Detector"
    
    # Allow custom repository via command line arguments
    if len(sys.argv) == 3:
        repo_owner = sys.argv[1]
        repo_name = sys.argv[2]
    elif len(sys.argv) > 1:
        print("Usage: python generate_invite_link.py [owner] [repo_name]")
        print("Example: python generate_invite_link.py mhmdelbadry1 GenImage-Detector")
        sys.exit(1)
    
    links = generate_invite_links(repo_owner, repo_name)
    print_links(links)
    
    # Additional tips
    print("📋 Quick Tips:")
    print("   • Share the repository URL on social media or messaging apps")
    print("   • Create an issue template for collaboration requests")
    print("   • Use GitHub's 'Invite collaborators' feature in Settings > Access")
    print("   • Consider creating a CONTRIBUTING.md file with guidelines")
    print()


if __name__ == "__main__":
    main()
