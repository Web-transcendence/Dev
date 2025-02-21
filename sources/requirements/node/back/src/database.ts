import Database from 'better-sqlite3';  // Assurez-vous que le module sqlite3 est installé
export const Client_db = new Database('client.db')  // Importation correcte de sqlite
import { z } from 'zod';
import { hashSync } from 'bcrypt';
import { FastifyRequest, FastifyReply } from 'fastify';

// Fonction pour initialiser la base de données
Client_db.exec(`
    CREATE TABLE IF NOT EXISTS Client (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
`);

console.log('Base de données initialisée avec succès.');

const clientSchema = z.object({
    name: z.string().min(2, "Le nom doit contenir au moins 2 caractères"),
    email: z.string().email("Email invalide"),
    password: z.string().min(6, "Le mot de passe doit contenir au moins 6 caractères"),
});

// Fonction pour créer un client dans la base de données
export const CreateClient = async (req: FastifyRequest, res: FastifyReply) => {
    try {
        const data = clientSchema.parse(req.body); // Valide les données
        const hashedPassword = hashSync(data.password, 10); //Hachage du mot de passe
        Client_db.prepare('INSERT INTO Client (name, email, password) VALUES (?, ?, ?)').run(
            data.name,
            data.email,
            hashedPassword
        );
        res.status(201).send({ message: 'Client créé avec succès' });
    } catch (error) {
        console.error('Erreur lors de l’insertion :', error);
        res.status(400).send({ error: 'Données invalides ou erreur serveur' });
    }
};

console.log(Client_db.prepare('SELECT * FROM Client').all());


// async (req, res) => {
//     const zParams = z.object({
//         name: z.string().min(1),
//         email: z.string().email(),
//         password: z.string().min(6), // Minimum 6 caractères pour le mot de passe
//     });
//     const {success, error, data} = zParams.safeParse(req.body);
//
//     if (!success) {
//         res.raw.writeHead(400);
//         res.raw.write(error);
//         res.raw.end();
//         return;
//     }
//
//     const {name, email, password} = data;
//
//     try {
//         await CreateClient(Client_db, {name, email, password}); // Enregistrement dans la DB
//         res.redirect('/register', 303); // Redirige vers la page d'inscription ou une autre page après inscription
//     } catch (err) {
//         res.raw.writeHead(500);
//         res.raw.write('Erreur lors de l\'enregistrement');
//         res.raw.end();
//     }