import Database from "better-sqlite3";
import bcrypt from "bcrypt";
import jwt from 'jsonwebtoken';

export const Client_db = new Database('client.db')  // Importation correcte de sqlite

Client_db.exec(`
    CREATE TABLE IF NOT EXISTS Client (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nickName TEXT NOT NULL,
        email UNIQUE NOT NULL COLLATE NOCASE,
        password TEXT NOT NULL,
        google_id INTEGER
    )
`);

export class User {
    id: string;

    constructor(id: string) {
        this.id = id;

        if (!Client_db.prepare("SELECT * FROM Client WHERE id = ?").get(this.id)) {
            throw new Error(`${this.id} not found`);
        }
    }

    static async addClient(nickName: string, email: string, password: string): Promise<{success: boolean, result: string}> {
        if (Client_db.prepare("SELECT * FROM Client WHERE nickName = ?").get(nickName)) {
            return {success: false, result: 'nickName'};
        }
        if (Client_db.prepare("SELECT * FROM Client WHERE email = ?").get(email)) {
            return {success: false, result: 'email'};
        }

        const hashedPassword = await bcrypt.hash(password, 10);

        const res = Client_db.prepare("INSERT INTO Client (nickName, email, password) VALUES (?, ?, ?)")
            .run(nickName, email, hashedPassword);
        if (res.changes === 0) {
            throw new Error(`User not inserted`);
        }
        const id : number | bigint = res.lastInsertRowid
        const stringId: string = id.toString();

        console.log("new user added");
        return {success: true, result: this.makeToken(stringId)};
    }

    static async login(nickName: string, password: string): Promise<{code: number, result: string}> {
        const userData = Client_db.prepare("SELECT id FROM Client WHERE nickName = ?").get(nickName) as {id: string};
        if (!userData)
            return {code: 409, result: "this nickName doesn't exist"};

        const client = new User(userData.id)

        if (!await client.isPasswordValid(password))
            return {code: 401, result: "invalid password"};
        return {code: 201, result: User.makeToken(client.id)}
    }

    async isPasswordValid(password: string): Promise<boolean> {
        const userData = Client_db.prepare("SELECT password FROM Client WHERE id = ?").get(this.id) as { password: string } | undefined;
        if (!userData) {
            throw new Error(`Database Error: cannot find password from ${this.id}`);
        }
        return (await bcrypt.compare(password, userData.password))
    }

    private static makeToken(id: string): string {
        const token = jwt.sign({id: id}, 'secret_key', {expiresIn: '1h'})
        return (token);
    }

    getProfile(): { nickName: string, email: string } {
        const userData = Client_db.prepare("SELECT nickName, email FROM Client WHERE id = ?").get(this.id) as { nickName: string, email: string } | undefined;
        if (!userData) {
            throw new Error(`Database Error: cannot find data from ${this.id}`);
        }
        return {nickName: userData.nickName, email: userData.email};
    }

}