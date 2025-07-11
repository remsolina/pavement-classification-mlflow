#!/usr/bin/env python3
"""
🚀 Local CI/CD Runner

This script runs the complete CI/CD pipeline locally using Act (GitHub Actions runner).
It provides a convenient way to test the entire pipeline before pushing to GitHub.

Prerequisites:
- Docker installed and running
- Act installed (https://github.com/nektos/act)

Usage:
    python run_local_ci.py                    # Run full pipeline
    python run_local_ci.py --job unit-tests   # Run specific job
    python run_local_ci.py --list             # List available jobs
    python run_local_ci.py --setup            # Setup Act and dependencies
"""

import subprocess
import sys
import os
import argparse
import json
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output."""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    BOLD = '\033[1m'
    NC = '\033[0m'  # No Color


class LocalCIRunner:
    """Local CI/CD pipeline runner using Act."""
    
    def __init__(self):
        self.workflow_file = ".github/workflows/local-ci.yml"
        self.act_config = ".actrc"
        
    def print_header(self):
        """Print runner header."""
        print(f"{Colors.BOLD}🚀 Local CI/CD Pipeline Runner{Colors.NC}")
        print("=" * 50)
        
    def print_success(self, message: str):
        """Print success message."""
        print(f"{Colors.GREEN}✅ {message}{Colors.NC}")
        
    def print_error(self, message: str):
        """Print error message."""
        print(f"{Colors.RED}❌ {message}{Colors.NC}")
        
    def print_warning(self, message: str):
        """Print warning message."""
        print(f"{Colors.YELLOW}⚠️  {message}{Colors.NC}")
        
    def print_info(self, message: str):
        """Print info message."""
        print(f"{Colors.BLUE}ℹ️  {message}{Colors.NC}")
    
    def check_prerequisites(self):
        """Check if prerequisites are installed."""
        self.print_info("Checking prerequisites...")
        
        # Check Docker
        try:
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                self.print_success(f"Docker: {result.stdout.strip()}")
            else:
                self.print_error("Docker not working properly")
                return False
        except FileNotFoundError:
            self.print_error("Docker not installed")
            return False
        
        # Check Act
        try:
            result = subprocess.run(['act', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                self.print_success(f"Act: {result.stdout.strip()}")
            else:
                self.print_error("Act not working properly")
                return False
        except FileNotFoundError:
            self.print_error("Act not installed")
            self.print_info("Install Act: https://github.com/nektos/act#installation")
            return False
        
        # Check workflow file
        if os.path.exists(self.workflow_file):
            self.print_success(f"Workflow file: {self.workflow_file}")
        else:
            self.print_error(f"Workflow file not found: {self.workflow_file}")
            return False
        
        return True
    
    def setup_act_config(self):
        """Setup Act configuration."""
        self.print_info("Setting up Act configuration...")
        
        act_config_content = """# Act configuration for local CI/CD
# Use medium-sized runner for better compatibility
-P ubuntu-latest=catthehacker/ubuntu:act-latest
-P ubuntu-20.04=catthehacker/ubuntu:act-20.04
-P ubuntu-18.04=catthehacker/ubuntu:act-18.04

# Environment variables
--env DOCKER_BUILDKIT=1
--env PYTHONPATH=/workspace

# Bind workspace
--bind
"""
        
        with open(self.act_config, 'w') as f:
            f.write(act_config_content)
        
        self.print_success(f"Act configuration created: {self.act_config}")
    
    def list_jobs(self):
        """List available jobs in the workflow."""
        self.print_info("Available jobs in the CI/CD pipeline:")
        
        try:
            result = subprocess.run(['act', '--list'], capture_output=True, text=True)
            if result.returncode == 0:
                print(result.stdout)
            else:
                self.print_error(f"Failed to list jobs: {result.stderr}")
        except Exception as e:
            self.print_error(f"Error listing jobs: {e}")
    
    def run_job(self, job_name=None, dry_run=False):
        """Run a specific job or the entire pipeline."""
        if job_name:
            self.print_info(f"Running job: {job_name}")
            cmd = ['act', '--job', job_name]
        else:
            self.print_info("Running complete CI/CD pipeline")
            cmd = ['act']
        
        if dry_run:
            cmd.append('--dry-run')
            self.print_info("Dry run mode - no actual execution")
        
        # Add verbose output
        cmd.extend(['--verbose', '--artifact-server-path', './artifacts'])
        
        try:
            self.print_info(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
            
            if result.returncode == 0:
                self.print_success("Pipeline completed successfully!")
            else:
                self.print_error(f"Pipeline failed with exit code {result.returncode}")
                
            return result.returncode == 0
            
        except KeyboardInterrupt:
            self.print_warning("Pipeline interrupted by user")
            return False
        except Exception as e:
            self.print_error(f"Error running pipeline: {e}")
            return False
    
    def run_tests_only(self):
        """Run only the test jobs."""
        self.print_info("Running test jobs only...")
        
        test_jobs = ['setup-validation', 'unit-tests', 'integration-tests']
        
        for job in test_jobs:
            self.print_info(f"Running {job}...")
            if not self.run_job(job):
                self.print_error(f"Job {job} failed")
                return False
        
        self.print_success("All test jobs completed successfully!")
        return True
    
    def setup_environment(self):
        """Setup local environment for CI/CD."""
        self.print_info("Setting up local CI/CD environment...")
        
        # Create artifacts directory
        artifacts_dir = Path("./artifacts")
        artifacts_dir.mkdir(exist_ok=True)
        self.print_success(f"Artifacts directory: {artifacts_dir}")
        
        # Setup Act configuration
        self.setup_act_config()
        
        # Create test data directory
        test_data_dir = Path("./local/test_data")
        if not test_data_dir.exists():
            test_data_dir.mkdir(parents=True)
            
            # Create sample test data structure
            classes = ['asphalt', 'chip-sealed', 'gravel']
            for class_name in classes:
                class_dir = test_data_dir / class_name
                class_dir.mkdir(exist_ok=True)
            
            self.print_success(f"Test data structure created: {test_data_dir}")
        
        self.print_success("Local CI/CD environment setup complete!")
    
    def clean_environment(self):
        """Clean up CI/CD artifacts and temporary files."""
        self.print_info("Cleaning up CI/CD environment...")
        
        cleanup_paths = [
            "./artifacts",
            "./local/test_data",
            "./.actrc",
            "./ci_report.md"
        ]
        
        for path in cleanup_paths:
            path_obj = Path(path)
            if path_obj.exists():
                if path_obj.is_dir():
                    import shutil
                    shutil.rmtree(path_obj)
                else:
                    path_obj.unlink()
                self.print_success(f"Removed: {path}")
        
        self.print_success("Cleanup complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='🚀 Local CI/CD Pipeline Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_local_ci.py                    # Run full pipeline
  python run_local_ci.py --job unit-tests   # Run specific job
  python run_local_ci.py --list             # List available jobs
  python run_local_ci.py --setup            # Setup environment
  python run_local_ci.py --tests-only       # Run only test jobs
  python run_local_ci.py --clean            # Clean up environment

Prerequisites:
  - Docker: https://docs.docker.com/get-docker/
  - Act: https://github.com/nektos/act#installation
        """
    )
    
    parser.add_argument('--job', '-j', help='Run specific job')
    parser.add_argument('--list', '-l', action='store_true', help='List available jobs')
    parser.add_argument('--setup', '-s', action='store_true', help='Setup CI/CD environment')
    parser.add_argument('--clean', '-c', action='store_true', help='Clean up environment')
    parser.add_argument('--tests-only', '-t', action='store_true', help='Run only test jobs')
    parser.add_argument('--dry-run', '-d', action='store_true', help='Dry run (no execution)')
    
    args = parser.parse_args()
    
    runner = LocalCIRunner()
    runner.print_header()
    
    # Handle specific commands
    if args.setup:
        runner.setup_environment()
        return 0
    
    if args.clean:
        runner.clean_environment()
        return 0
    
    if args.list:
        runner.list_jobs()
        return 0
    
    # Check prerequisites
    if not runner.check_prerequisites():
        runner.print_error("Prerequisites not met. Run with --setup to configure environment.")
        return 1
    
    # Run pipeline
    success = False
    
    if args.tests_only:
        success = runner.run_tests_only()
    elif args.job:
        success = runner.run_job(args.job, dry_run=args.dry_run)
    else:
        success = runner.run_job(dry_run=args.dry_run)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
