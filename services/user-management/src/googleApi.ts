import {FastifyReply, FastifyRequest} from "fastify";
import { OAuth2Client } from 'google-auth-library';
import { Client_db} from "./User.js";
import jwt from "jsonwebtoken";

// Initialisation du client Google OAuth2
const client = new OAuth2Client("562995219569-0icrl4jh4ku3h312qmjm8ek57fqt7fp5.apps.googleusercontent.com");

export async function googleAuth(request: FastifyRequest, reply: FastifyReply):Promise<void> {
    const { credential } = request.body as { credential: string };
    try {
        // Vérification du jeton Google
        const ticket = await client.verifyIdToken({
            idToken: credential,
            audience: "562995219569-0icrl4jh4ku3h312qmjm8ek57fqt7fp5.apps.googleusercontent.com",
        });

        // Récupération des informations de l'utilisateur
        const payload = ticket.getPayload();
        const userId = payload?.sub; // ID unique de l'utilisateur

        if (!payload || !userId) {
            throw new Error('Invalid payload from Google');
        }

        // Log des informations de l'utilisateur
        console.log('User ID:', userId);
        console.log('Email:', payload.email);
        console.log('Name:', payload.name);
        console.log('Profile Picture URL:', payload.picture);  // L'URL de la photo de profil

        if (Client_db.prepare("SELECT * FROM Client WHERE email = ?").get(payload.email))
            console.log('Email Already Register:', payload.email);
        else {
            const res = Client_db.prepare("INSERT INTO Client (name, email, password, google_id) VALUES (?, ?, ?, ?)")
                .run(payload.name, payload.email, 'NOTGIVEN', userId);
            console.log('New Email:', payload.email);
        }
            const rows = Client_db.prepare(`SELECT * FROM Client`).all(); // Print clients info
            console.table(rows);
        const token = jwt.sign({ name: payload.name, email: payload.email, avatar: payload.picture, userId: payload.sub }, 'secret_key', { expiresIn: '1h' });
        return reply.send({token, valid: true, name: payload.name, avatar: payload.picture});
    } catch (error) {
        console.log('Error verifying Google token:', error);
        reply.status(400).send({ success: false, error: 'Invalid token' });
    }
}