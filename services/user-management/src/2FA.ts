import { authenticator } from 'otplib';
import * as qrcode from 'qrcode';

export async function generateTOTP(user: string) {
    const secret = authenticator.generateSecret();
    const otpauthUri = authenticator.keyuri(user, 'MyApp', secret);
    const qrCode = await qrcode.toDataURL(otpauthUri);
    return { secret, qrCode };
}

export function verifyTOTP(secret: string, token: string): boolean {
    return authenticator.verify({ token, secret });
}
