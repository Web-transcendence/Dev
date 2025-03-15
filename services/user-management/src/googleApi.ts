import {FastifyReply, FastifyRequest} from "fastify";
import { OAuth2Client } from 'google-auth-library';

// Initialisation du client Google OAuth2
const client = new OAuth2Client('/auth/google');

export async function googleAuth(req: FastifyRequest, res: FastifyReply):Promise<void> {
    const { credential } = request.body as { credential: string };
    try {
        // Vérification du jeton Google
        const ticket = await client.verifyIdToken({
            idToken: credential,
            audience: '/auth/google',
        });

        // Récupération des informations de l'utilisateur
        const payload = ticket.getPayload();
        const userId = payload?.sub; // ID unique de l'utilisateur

        if (!payload || !userId) {
            throw new Error('Invalid payload from Google');
        }

        // Log des informations de l'utilisateur
        app.log.info('User ID:', userId);
        app.log.info('Email:', payload.email);
        app.log.info('Name:', payload.name);

        // Réponse réussie
        reply.status(200).send({ success: true, user: payload });
    } catch (error) {
        app.log.error('Error verifying Google token:', error);
        reply.status(400).send({ success: false, error: 'Invalid token' });
    }
}