import Database from 'better-sqlite3';  // Assurez-vous que le module sqlite3 est installé
export const Client_db = new Database('client.db')  // Importation correcte de sqlite
// import { z } from 'zod';
// import fastify from "fastify";
// import bcrypt from "bcrypt";

// Activation du parsing JSON

// Fonction pour initialiser la base de données
// Client_db.exec(`
//     CREATE TABLE IF NOT EXISTS Client (
//         id INTEGER PRIMARY KEY AUTOINCREMENT,
//         name TEXT NOT NULL,
//         email UNIQUE NOT NULL COLLATE NOCASE,
//         password TEXT NOT NULL
//     )
// `);
//
// console.log('Base de données initialisée avec succès.');
//
// const clientSchema = z.object({
//     name: z.string().min(2, "Le nom doit contenir au moins 2 caractères"),
//     email: z.string().email("Email invalide"),
//     password: z.string().min(6, "Le mot de passe doit contenir au moins 6 caractères"),
// });

// async function CreateClient(name: string, email: string, password: string): Promise<void> {
//     if (!name || !email || !password) {
//         console.error("Erreur : Toutes les informations sont requises !");
//         return;
//     }
//
//     try {
//         const response = await fetch("http://localhost:3000/register", {
//             method: "POST",
//             headers: { "Content-Type": "application/json" },
//             body: JSON.stringify({ name: name, email: email, password: password }), // Version explicite
//         });
//
//         if (!response.ok) {
//             throw new Error(`Erreur ${response.status}: ${response.statusText}`);
//         }
//
//         const data: { message: string; user?: any; error?: string } = await response.json();
//         console.log(data);
//     } catch (error) {
//         console.error("Erreur lors de l'inscription:", error);
//     }
// }
// registerUser("Alice", "alice@example.com", "secure123");

    //     console.log('Creating client...');
//     try {
//         if (!req.body) {
//             return res.status(422).send({error: "Le corps de la requête est vide."});
//         }
//         // Validation des données avec Zod
//     console.log('Creating client...1');
//         const data = clientSchema.parse(req.body);
//
//     console.log('Creating client...2');
//         // Hachage du mot de passe
//         const hashedPassword = hashSync(data.password, 10);
//
//     console.log('Creating client...3');
//         // Exécution de la requête d'insertion
//         const stmt = Client_db.prepare('INSERT INTO Client (name, email, password) VALUES (?, ?, ?)');
//         stmt.run(data.name, data.email, hashedPassword);
//     console.log('Creating client...4');
//
//         res.status(201).send({ message: 'Client créé avec succès' });
//
//     } catch (error) {
//         console.error('Erreur lors de l’insertion :', error);
//
//         // Vérifier si l'erreur est une erreur SQLite
//         if (error instanceof Error && 'code' in error && error.code === 'SQLITE_CONSTRAINT') {
//             res.status(488).send({ error: 'Cet email est déjà utilisé' });
//
//             // Vérifier si l'erreur est une erreur de validation Zod
//         } else if (error instanceof Error && error.name === 'ZodError') {
//             res.status(477).send({ error: 'Données invalides', details: (error as any).errors });
//
//         } else {
//             res.status(500).send({ error: 'Erreur serveur' });
//         }
//     }
// };



// Fonction pour créer un client dans la base de données
// export const CreateClient(req: FastifyRequest, res: FastifyReply) => {
//     try {
//         const data = clientSchema.parse(req.body); // Valide les données
//         const hashedPassword = hashSync(data.password, 10); //Hachage du mot de passe
//         Client_db.prepare('INSERT INTO Client ( name, email, password) VALUES (?, ?, ?)').run(
//             data.name,
//             data.email,
//             hashedPassword
//         );
//         console.log(Client_db.prepare('SELECT * FROM Client').all());
//         res.status(201).send({ message: 'Client créé avec succès' });
//     } catch (error) {
//         if (error.code === 'SQLITE_CONSTRAINT') {
//             res.status(400).send({error: 'Cet email est déjà utilisé'});
//         } else {
//             res.status(500).send({error: 'Erreur serveur'});
//         }
//     }
// };

//
// export function clearDatabase() {
//     const tables: { name: string }[] = Client_db.prepare(
//         "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
//     ).all();
//     Client_db.transaction(() => {
//         tables.forEach((table) => {
//             const tableName: string = table.name;
//             Client_db.prepare(`DELETE FROM ${tableName}`).run();
//             Client_db.prepare(`VACUUM`).run(); // Optional: Reclaims space
//         });
//     })();
//
//     console.log("All tables have been cleared!");
// }
