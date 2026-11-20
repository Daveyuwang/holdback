# Security

holdback exists to keep private values on your machine, so a value that leaves it is a security bug.

## Reporting

Report a leak, a way around the redaction, or a way to read values from the local process through GitHub's
private vulnerability reporting on this repository ("Report a vulnerability" under the Security tab). Do not
open a public issue for these.

Include the holdback version (`holdback --version`), the mode (proxy or hook), the agent and its version, and
the smallest input that shows the problem. Use made-up values. `holdback review --masked` output and, in hook
mode, the lines of `sightings.log` hold only shapes and are safe to attach.

## What counts

- A value that holdback detected but that still reached the provider.
- A way for a web page or another user on the machine to read values from the local process.
- A way for content the agent reads to make holdback restore a value where it should not.

A value the detector does not recognise is a known limit, listed in the README, and belongs in a normal issue.
The channels hook mode does not cover are listed in the README as well.
