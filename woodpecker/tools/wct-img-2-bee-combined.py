#!/usr/bin/env python
# Convert ms-active and ms-masked cluster tarballs to bee format.
# Both go into data/0/ as 0-apa<N>-active.json / 0-apa<N>-masked.json, i.e. ONE
# bee event in which "active" and "masked" appear as two selectable algorithms.
# (An earlier version of this comment claimed masked went to data/1/; it never
# did.)
#
# Accepts any number of active and masked files (not limited to 8 anodes).
# The anode index is parsed from the filename (clusters-apa-anode<N>-ms-*.tar.gz).
#
# Usage:
#   python wct-img-2-bee-combined.py \
#     [--active clusters-apa-anode0-ms-active.tar.gz ...] \
#     [--masked clusters-apa-anode0-ms-masked.tar.gz ...]
#
#   Or positional (all active first, then all masked), with --split N:
#   python wct-img-2-bee-combined.py --split 4 \
#     a0-active.tar.gz a1-active.tar.gz a2-active.tar.gz a3-active.tar.gz \
#     a0-masked.tar.gz a1-masked.tar.gz a2-masked.tar.gz a3-masked.tar.gz

import argparse
import os
import re
import sys

# anode index -> (speed, x0) for ProtoDUNE-VD
_ANODE_PARAMS = {
    0: ('-1.56*mm/us', '-341.5*cm'),
    1: ('-1.56*mm/us', '-341.5*cm'),
    2: ('-1.56*mm/us', '-341.5*cm'),
    3: ('-1.56*mm/us', '-341.5*cm'),
    4: ( '1.56*mm/us',  '341.5*cm'),
    5: ( '1.56*mm/us',  '341.5*cm'),
    6: ( '1.56*mm/us',  '341.5*cm'),
    7: ( '1.56*mm/us',  '341.5*cm'),
}
_DEFAULT_PARAMS = ('-1.56*mm/us', '-341.5*cm')

_ANODE_RE = re.compile(r'anode(\d+)', re.IGNORECASE)


def _anode_idx(filepath):
    """Extract anode index from filename, e.g. clusters-apa-anode3-ms-active.tar.gz -> 3."""
    m = _ANODE_RE.search(os.path.basename(filepath))
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot parse anode index from filename: {filepath}")


def bee_blobs(tarball, outfile, anode_idx, density=1):
    """Convert one cluster tarball to a bee JSON.  Returns True on success.

    The return code MUST be checked by the caller: this used to be a bare
    os.system() whose status was discarded, so when every masked file failed
    (KeyError: 'measure' -- the geometry-only masked fork has no measure nodes)
    the script still went on to zip up an active-only upload and print a
    perfectly healthy-looking bee URL.  Half the output vanished silently.
    """
    speed, x0 = _ANODE_PARAMS.get(anode_idx, _DEFAULT_PARAMS)
    cmd = ('wirecell-img bee-blobs -g protodunevd -s uniform -d %f'
           ' --speed "%s" --t0 "0*us" --x0 "%s"'
           ' -o %s %s') % (density, speed, x0, outfile, tarball)
    print(cmd)
    rc = os.system(cmd)
    rc = os.waitstatus_to_exitcode(rc) if os.name == 'posix' else rc
    if rc != 0:
        print('ERROR: bee-blobs failed (exit %d) for %s' % (rc, tarball),
              file=sys.stderr)
        return False
    if not os.path.exists(outfile):
        print('ERROR: bee-blobs reported success but wrote no %s' % outfile,
              file=sys.stderr)
        return False
    return True


def main(active_files, masked_files):
    if os.path.exists('data/0'):
        print('found old data, removing ...')
        os.system('rm -rf data')
    if os.path.exists('upload.zip'):
        os.system('rm -f upload.zip')
    os.system('mkdir -p data/0')

    failed = []
    for kind, files in (('active', active_files), ('masked', masked_files)):
        for fp in files:
            idx = _anode_idx(fp)
            out = 'data/0/0-apa%d-%s.json' % (idx, kind)
            if not bee_blobs(fp, out, idx):
                failed.append(fp)

    if failed:
        print('\nERROR: %d of %d cluster file(s) failed to convert:'
              % (len(failed), len(active_files) + len(masked_files)),
              file=sys.stderr)
        for fp in failed:
            print('  ' + os.path.basename(fp), file=sys.stderr)
        print('Refusing to upload a partial set -- fix the conversion or drop\n'
              'the offending files explicitly.', file=sys.stderr)
        return 1

    os.system('zip -r upload data')
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert WCT imaging cluster tarballs to bee display format")
    parser.add_argument(
        '--active', nargs='+', metavar='FILE', default=[],
        help='ms-active cluster tar.gz files (any number of anodes)')
    parser.add_argument(
        '--masked', nargs='+', metavar='FILE', default=[],
        help='ms-masked cluster tar.gz files (any number of anodes)')
    parser.add_argument(
        '--split', type=int, default=None, metavar='N',
        help='Split positional args: first N are active, rest are masked')
    parser.add_argument(
        'files', nargs='*', metavar='FILE',
        help='Positional files (use --split N to divide active/masked)')
    args = parser.parse_args()

    active_files = list(args.active)
    masked_files = list(args.masked)

    if args.files:
        if args.split is not None:
            active_files += args.files[:args.split]
            masked_files += args.files[args.split:]
        else:
            # Guess by filename
            for f in args.files:
                if 'masked' in f:
                    masked_files.append(f)
                else:
                    active_files.append(f)

    if not active_files and not masked_files:
        parser.print_help()
        sys.exit(1)

    sys.exit(main(active_files, masked_files))
