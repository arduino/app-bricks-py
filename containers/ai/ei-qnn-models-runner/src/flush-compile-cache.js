// SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
//
// SPDX-License-Identifier: MPL-2.0

// Persist the V8 compile cache when the runner is stopped: node writes it only on a normal exit,
// and the runner is stopped with SIGTERM, which it does not handle.
process.on("SIGTERM", () => {
  require("node:module").flushCompileCache();
  process.exit(0);
});
