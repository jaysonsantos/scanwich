# Third-party notices

Scanwich's source code is licensed under the MIT License. Its dependencies retain their
own licenses and are not relicensed by Scanwich.

The direct dependencies shipped by the Nix package and container include:

| Component | License |
| --- | --- |
| EasyOCR | Apache-2.0 |
| Pillow | MIT-CMU |
| ReportLab | BSD-3-Clause |
| ImageMagick | Apache-2.0 |
| Ghostscript | AGPL-3.0-or-later |

The test-only pypdf dependency is BSD-3-Clause. Transitive dependencies retain their
respective licenses as recorded by the pinned Nixpkgs revision in `flake.lock`.

The container redistributes an unmodified Ghostscript binary as a separate program used
by ImageMagick. Its exact Nix-pinned source is included in the image at
`/opt/third-party-sources/ghostscript`. Ghostscript remains licensed under the GNU Affero
General Public License; see <https://ghostscript.com/licensing/>.
