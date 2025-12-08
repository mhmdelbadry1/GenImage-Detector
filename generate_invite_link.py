#!/usr/bin/env python3
"""
GitHub Repository Invite Link Generator

This script helps you generate shareable links for inviting collaborators
to your GitHub repository without needing to ask for their usernames individually.
"""

import sys
import re


def validate_github_name(name, name_type):
    """
    Validate GitHub username or repository name.
    
    Args:
        name (str): The name to validate
        name_type (str): Type of name being validated ('owner' or 'repository')
    
    Returns:
        bool: True if valid, False otherwise
    
    Raises:
        ValueError: If the name is invalid
    """
    # GitHub usernames and repo names can contain alphanumeric characters, hyphens, and underscores
    # They cannot start with a hyphen and have length restrictions
    if not name or len(name) > 100:
        raise ValueError(f"GitHub {name_type} name must be between 1 and 100 characters")
    
    # Check for valid characters (alphanumeric, hyphens, underscores, and dots for repos)
    if not re.match(r'^[a-zA-Z0-9._-]+$', name):
        raise ValueError(f"GitHub {name_type} name can only contain alphanumeric characters, hyphens, underscores, and dots")
    
    # Cannot start with a hyphen
    if name.startswith('-'):
        raise ValueError(f"GitHub {name_type} name cannot start with a hyphen")
    
    return True


def generate_invite_links(repo_owner, repo_name):
    """
    Generate various shareable links for a GitHub repository.
    
    Args:
        repo_owner (str): The GitHub username or organization name
        repo_name (str): The repository name
    
    Returns:
        dict: Dictionary containing different types of links with the following keys:
            - 'repository': Main repository URL
            - 'issues': Repository issues page URL
            - 'discussions': Repository discussions page URL
            - 'fork': URL to fork the repository
            - 'settings_access': Repository access settings URL
    
    Raises:
        ValueError: If repo_owner or repo_name contain invalid characters
    """
    # Validate inputs
    validate_github_name(repo_owner, "owner")
    validate_github_name(repo_name, "repository")
    
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
    # Expected: script_name owner repo_name (3 arguments total)
    if len(sys.argv) == 3:
        repo_owner = sys.argv[1]
        repo_name = sys.argv[2]
    elif len(sys.argv) != 1:
        # Any other number of arguments (2, 4, 5, etc.) shows usage
        print("Usage: python generate_invite_link.py [owner] [repo_name]")
        print("Example: python generate_invite_link.py mhmdelbadry1 GenImage-Detector")
        print("\nNote: Both owner and repo_name must be provided together, or neither for defaults.")
        sys.exit(1)
    
    try:
        links = generate_invite_links(repo_owner, repo_name)
        print_links(links)
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("Please check that the repository owner and name are valid GitHub identifiers.")
        sys.exit(1)
    
    # Additional tips
    print("📋 Quick Tips:")
    print("   • Share the repository URL on social media or messaging apps")
    print("   • Create an issue template for collaboration requests")
    print("   • Use GitHub's 'Invite collaborators' feature in Settings > Access")
    print("   • Consider creating a CONTRIBUTING.md file with guidelines")
    print()


if __name__ == "__main__":
    main()
