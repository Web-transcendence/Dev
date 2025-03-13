import Database from "better-sqlite3";
import {FastifyRequest} from "fastify";
import bcrypt from "bcrypt";
import {Hash} from "node:crypto";

type UserStatus = "Exists" | "NotExists";


export const Client_db = new Database('client.db')  // Importation correcte de sqlite




Client_db.exec(`
    CREATE TABLE IF NOT EXISTS Client (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email UNIQUE NOT NULL COLLATE NOCASE,
        password TEXT NOT NULL
    )
`);


export class User {
    name: string;
    private status: UserStatus;

    constructor(name: string) {
        this.name = name;
        this.status = "NotExists";
        if (Client_db.prepare("SELECT * FROM Client WHERE name = ?").get(this.name)) {
            this.status = "Exists";
        }
    }

    async addClient(email: string, password: string) {
        if (this.status === "Exists") {
            return {success: false, errorType: 'name'};
        }
        if (Client_db.prepare("SELECT * FROM Client WHERE email = ?").get(email)) {
            return {success: false, errorType: 'email'};
        }

        const hashedPassword = await bcrypt.hash(password, 10);

        const res = Client_db.prepare("INSERT INTO Client (name, email, password) VALUES (?, ?, ?)")
            .run(this.name, email, hashedPassword);
        if (res.changes === 0) {
            throw new Error(`User not inserted`);
        }
        this.status = "Exists";
        console.log('user added to client');
        return { success: true, id: res.lastInsertRowid };
    }


}