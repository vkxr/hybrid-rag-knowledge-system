import { getDb, migrate } from "../lib/db";

await migrate(getDb());
console.log("Schema is up to date.");
process.exit(0);
