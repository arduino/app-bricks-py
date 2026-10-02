#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# Mirror the GitHub releases (tag, notes, assets, flags) and the numeric-tagged container
# images of SOURCE_REPO into a target repo of another owner. Incremental and rerun-safe:
# matching items are skipped, differing ones overwritten, extra target items kept. The
# source is only read. Actions on the target is paused while tags are created.
#
# Usage: mirror_releases.sh <owner/repo>   [SOURCE_REPO=arduino/app-bricks-py] [PACKAGE_PREFIX=app-bricks/]
# Needs bash 3.2+, crane, and gh logged in with repo, workflow, write:packages, delete:packages.
set -euo pipefail
shopt -s nullglob

TARGET_REPO="${1:?usage: $0 <owner/repo>}"
SOURCE_REPO="${SOURCE_REPO:-arduino/app-bricks-py}"
PACKAGE_PREFIX="${PACKAGE_PREFIX:-app-bricks/}"

SRC_OWNER="${SOURCE_REPO%%/*}"
DST_OWNER="${TARGET_REPO%%/*}"

# The source is only ever read: refuse any target that could alias it.
lower() { tr '[:upper:]' '[:lower:]' <<<"$1"; }
if [[ "$(lower "$DST_OWNER")" == "$(lower "$SRC_OWNER")" ]]; then
  echo "error: target '$TARGET_REPO' must not belong to the source owner '$SRC_OWNER'" >&2
  exit 1
fi
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

# --- 1. Images: every container package of the source owner under PACKAGE_PREFIX,
# numeric tags only. Matching digests are skipped, crane skips blobs already present. ---
echo "## Images -> ghcr.io/$DST_OWNER/$PACKAGE_PREFIX"
owner_kind=$([[ "$(gh api "users/$SRC_OWNER" --jq .type)" == Organization ]] && echo orgs || echo users)
packages=$(gh api "$owner_kind/$SRC_OWNER/packages?package_type=container&per_page=100" --paginate \
  --jq ".[].name | select(startswith(\"$PACKAGE_PREFIX\"))" | sort)
crane auth login ghcr.io -u "$DST_OWNER" -p "$(gh auth token)"
for pkg in $packages; do
  src="ghcr.io/$SRC_OWNER/$pkg"
  dst="ghcr.io/$DST_OWNER/$pkg"
  for tag in $(crane ls "$src" | grep -E '^[0-9]' || true); do
    if [[ "$(crane digest "$src:$tag")" == "$(crane digest "$dst:$tag" 2>/dev/null)" ]]; then
      continue
    fi
    echo "  $pkg:$tag"
    crane copy -j 4 "$src:$tag" "$dst:$tag"
  done
done

# --- 2. Actions on the target are switched off before the first tag is created,
# so no workflow starts from the pushed tags, and restored on exit. ---
ACTIONS_ENABLED=""
actions_off() {
  [[ -n "$ACTIONS_ENABLED" ]] && return
  ACTIONS_ENABLED=$(gh api "repos/$TARGET_REPO/actions/permissions" --jq .enabled)
  gh api -X PUT "repos/$TARGET_REPO/actions/permissions" -F enabled=false >/dev/null
  echo "## Actions disabled on $TARGET_REPO (was enabled=$ACTIONS_ENABLED)"
}
restore_actions() {
  if [[ -n "$ACTIONS_ENABLED" ]]; then
    gh api -X PUT "repos/$TARGET_REPO/actions/permissions" -F enabled="$ACTIONS_ENABLED" >/dev/null
    echo "## Actions restored on $TARGET_REPO (enabled=$ACTIONS_ENABLED)"
  fi
  rm -rf "$WORKDIR"
}
trap restore_actions EXIT

# --- 3. Releases, oldest first. A release is left alone when tag commit, title,
# notes, flags and assets all match; otherwise it is recreated from the source. ---
FIELDS='[.name, .isPrerelease, .isDraft, (.body | sub("\n+$"; "")), (.assets | map("\(.name):\(.size)") | sort | join(","))] | @json'
fingerprint() {  # <repo> <tag>: release fields plus the commit the tag points to, empty if absent
  local repo=$1 tag=$2
  gh release view "$tag" -R "$repo" --json name,isPrerelease,isDraft,body,assets --jq "$FIELDS" 2>/dev/null || return 0
  gh api "repos/$repo/commits/$tag" --jq .sha
}

# --jq runs per page, so releases are oldest first within each page of 100.
gh api "repos/$SOURCE_REPO/releases?per_page=100" --paginate --jq 'reverse | .[].tag_name' > "$WORKDIR/tags"
LATEST_TAG=$(gh release view -R "$SOURCE_REPO" --json tagName --jq .tagName)

echo "## Releases -> $TARGET_REPO"
for tag in $(cat "$WORKDIR/tags"); do
  if [[ "$(fingerprint "$SOURCE_REPO" "$tag")" == "$(fingerprint "$TARGET_REPO" "$tag")" ]]; then
    continue
  fi
  echo "  $tag"
  dir="$WORKDIR/assets/$tag"
  mkdir -p "$dir"
  gh release download "$tag" -R "$SOURCE_REPO" -D "$dir" 2>/dev/null || true  # a release may have no assets
  printf '%s' "$(gh release view "$tag" -R "$SOURCE_REPO" --json body --jq .body)" > "$dir.notes"
  read -r prerelease draft name <<<"$(gh release view "$tag" -R "$SOURCE_REPO" --json name,isPrerelease,isDraft --jq '[.isPrerelease, .isDraft, .name] | @tsv')"
  sha=$(gh api "repos/$SOURCE_REPO/commits/$tag" --jq .sha)

  actions_off
  # A tag without a release is not removed by --cleanup-tag: drop it when it points
  # elsewhere, so the release recreates it on the source commit.
  if ! gh release view "$tag" -R "$TARGET_REPO" >/dev/null 2>&1 \
     && [[ "$(gh api "repos/$TARGET_REPO/commits/$tag" --jq .sha 2>/dev/null)" != "$sha" ]] \
     && gh api "repos/$TARGET_REPO/git/ref/tags/$tag" >/dev/null 2>&1; then
    gh api -X DELETE "repos/$TARGET_REPO/git/refs/tags/$tag" >/dev/null
  fi
  gh release delete "$tag" -R "$TARGET_REPO" --cleanup-tag --yes 2>/dev/null || true

  flags=(--target "$sha" --title "$name" --notes-file "$dir.notes" --latest="$([[ $tag == "$LATEST_TAG" ]] && echo true || echo false)")
  [[ $prerelease == true ]] && flags+=(--prerelease)
  [[ $draft == true ]] && flags+=(--draft)
  gh release create "$tag" -R "$TARGET_REPO" "${flags[@]}" "$dir"/*
done

# The "Latest" marker follows the source too.
if [[ "$(gh release view -R "$TARGET_REPO" --json tagName --jq .tagName 2>/dev/null)" != "$LATEST_TAG" ]]; then
  echo "  marking $LATEST_TAG as latest"
  gh release edit "$LATEST_TAG" -R "$TARGET_REPO" --latest
fi

echo "## Done."
