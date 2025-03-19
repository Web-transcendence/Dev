// import { FastifyInstance } from 'fastify';
// import speakeasy from 'speakeasy';
// import qrcode from 'qrcode';
//
// // In-memory user store (for demo purposes)
// const users = new Map<string, { secret: string; enabled2fa: boolean }>();
//
//
// export async function twoFa(fastify: FastifyInstance):Promise<void> {
//     fastify.post('/setup-2fa', async (request, reply) => {
//         const {username} = request.body as { username: string };
//         if (!username) {
//             return reply.status(400).send({message: 'Username is required'});
//         }
//         // Generate a TOTP secret for the user
//         const secret = speakeasy.generateSecret({name: username});
//
//         // Save the secret (for demo, we're using in-memory storage)
//         users.set(username, {secret: secret.base32, enabled2fa: false});
//
//         // Generate a QR code for user to scan in their authenticator app
//         const qrCodeUrl = await qrcode.toDataURL(secret.otpauth_url);
//
//         return reply.send({qrCodeUrl, message: 'Scan the QR code with your authenticator app.'});
//     });
//
//     // Route to verify the TOTP token from the authenticator app
//     fastify.post('/verify-2fa', async (request, reply) => {
//         const {username, token} = request.body as { username: string; token: string };
//         const user = users.get(username);
//
//         if (!user) {
//             return reply.status(400).send({message: 'User not found'});
//         }
//
//         // Verify the token with the user's stored secret
//         const isVerified = speakeasy.totp.verify({
//             secret: user.secret,
//             encoding: 'base32',
//             token,
//         });
//
//         if (isVerified) {
//             // Mark 2FA as enabled for the user
//             user.enabled2fa = true;
//             return reply.send({message: '2FA setup successful'});
//         } else {
//             return reply.status(400).send({message: 'Invalid token'});
//         }
//     });
//
//     // Route to check 2FA status
//     fastify.get('/check-2fa-status', async (request, reply) => {
//         const {username} = request.query as { username: string };
//         const user = users.get(username);
//
//         if (!user) {
//             return reply.status(400).send({message: 'User not found'});
//         }
//
//         return reply.send({enabled2fa: user.enabled2fa});
//     });
// }


// import { authenticator } from 'otplib';
// import * as qrcode from 'qrcode';
//
// export async function generateTOTP(user: string) {
//     const secret = authenticator.generateSecret();
//     const otpauthUri = authenticator.keyuri(user, 'MyApp', secret);
//     const qrCode = await qrcode.toDataURL(otpauthUri);
//     return { secret, qrCode };
// }
//
// export function verifyTOTP(secret: string, token: string): boolean {
//     return authenticator.verify({ token, secret });
// }
