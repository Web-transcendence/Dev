import Database from "better-sqlite3";
import {FastifyReply, FastifyRequest} from "fastify";

export const Client_db = new Database('client.db')  // Importation correcte de sqlite




Client_db.exec(`
    CREATE TABLE IF NOT EXISTS Client (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email UNIQUE NOT NULL COLLATE NOCASE,
        password TEXT NOT NULL
    )
`);



export function emailExist(email: string) {
    const existingClient = Client_db.prepare("SELECT * FROM Client WHERE email = ?").get(email);
    return !!existingClient;
}


export async function createClient(req: FastifyRequest, res: FastifyReply) {
    const { name, email, password } = req.body as { name: string; email: string; password: string };


    if (!name || !email || !password) {
        return res.status(400).send({error: "Toutes les informations sont requises !"});
    }

    const insert = Client_db.prepare("INSERT INTO Client (name, email, password) VALUES (?, ?, ?)");
    const result = insert.run(name, email, password);
    // language=SQL format=false
    Client_db.prepare(`SELECT * FROM Client`).all();
    if (result.changes === 1) {
        Client_db.prepare("SELECT * FROM Client");
        const newClientId = result.lastInsertRowid;
        const newClient = Client_db.prepare("SELECT name FROM Client WHERE id = ?").get(newClientId) as { name: string } | undefined;
        if (newClient) {
            console.log("Nouveau client enregistré:", newClient.name);
        }
        return res.redirect(`/part/login?name=${encodeURIComponent(newClient?.name || "Utilisateur")}`);
        // return reply.redirect(`/part/login`);
        // return reply.status(201).send({ message: "Compte client créé avec succès", id: result.lastInsertRowid });
    } else {
        return res.status(500).send({ error: "Erreur lors de la création du compte" });
    }
}