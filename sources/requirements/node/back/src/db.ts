
// const db = new sqlite3.Database(':memory:');
//
// db.serialize(() => {
//     db.run("CREATE TABLE lorem (info TEXT)");
//
//     const stmt = db.prepare("INSERT INTO lorem VALUES (?)");
//     for (let i = 0; i < 10; i++) {
//         stmt.run("Ipsum " + i);
//     }
//     stmt.finalize();
// });
//
// export {db};

import sqlite3 from 'sqlite3';  // Assurez-vous que le module sqlite3 est installé
import { open, Database } from 'sqlite';  // Importation correcte de sqlite

export interface Client {
    name: string;
    email: string;
    password: string;
}

export interface message {
    email: string;
    password: string;
    msg: string;
}

// Fonction pour initialiser la base de données
export async function initDb(): Promise<Database> {
    const db = await open({
        filename: './database.db',
        driver: sqlite3.Database,
    });

    // Créer la table si elle n'existe pas
    await db.run(`
    CREATE TABLE IF NOT EXISTS clients (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL,
      email TEXT NOT NULL UNIQUE,
      password TEXT NOT NULL
    )
  `);

    return db;
}

// Fonction pour créer un client dans la base de données
export async function createClient(
    db: Database,
    { name, email, password }: Client
): Promise<void> {
    const stmt = await db.prepare('INSERT INTO clients (name, email, password) VALUES (?, ?, ?)');
    await stmt.run(name, email, password);
    await stmt.finalize();
}

