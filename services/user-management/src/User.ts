import Database from "better-sqlite3";
import bcrypt from "bcrypt";
import jwt from 'jsonwebtoken';

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

    async addClient(email: string, password: string): Promise<{success: boolean, result: string}> {
        if (this.status === "Exists") {
            return {success: false, result: 'name'};
        }
        if (Client_db.prepare("SELECT * FROM Client WHERE email = ?").get(email)) {
            return {success: false, result: 'email'};
        }

        const hashedPassword = await bcrypt.hash(password, 10);

        const res = Client_db.prepare("INSERT INTO Client (name, email, password) VALUES (?, ?, ?)")
            .run(this.name, email, hashedPassword);
        if (res.changes === 0) {
            throw new Error(`User not inserted`);
        }

        const rows = Client_db.prepare(`SELECT * FROM Client`).all(); // Print clients info
        console.table(rows);

        this.status = "Exists";
        console.log("new user added");
        return {success: true, result: this.makeToken()};
    }

    async isPasswordValid(password: string): Promise<boolean> {
        const userData = Client_db.prepare("SELECT password FROM Client WHERE name = ?").get(this.name) as { password: string } | undefined;
        if (!userData) {
            throw new Error(`Database Error: cannot find password from ${this.name}`);
        }
        return (await bcrypt.compare(password, userData.password))
    }

    makeToken(): string {
        const token = jwt.sign({id: 1, name: this.name}, 'secret_key', {expiresIn: '1h'})
        return (token);
    }

    getStatus() {
        return this.status;
    }
}