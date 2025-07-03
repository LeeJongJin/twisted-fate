#!/usr/bin/env node

/* eslint-disable @typescript-eslint/no-var-requires */
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

/**
 * 현재 설치된 Cursor 확장 프로그램을 내보내는 스크립트
 * cursor-extensions.txt 파일에 확장 프로그램 목록을 저장
 */
async function exportExtensions() {
  try {
    console.log('🔍 현재 설치된 Cursor 확장 프로그램을 확인합니다...');

    // 현재 설치된 확장 프로그램 목록 가져오기
    const extensionsList = execSync('cursor --list-extensions', {
      encoding: 'utf8',
      timeout: 10000, // 10초 타임아웃
    }).trim();

    if (!extensionsList) {
      console.log('⚠️  설치된 확장 프로그램이 없습니다.');
      return;
    }

    // 확장 프로그램 목록을 배열로 변환
    const extensions = extensionsList.split('\n').filter((line) => line.trim());

    console.log(`📦 ${extensions.length}개의 확장 프로그램을 발견했습니다.`);

    // 환경 폴더 생성 (존재하지 않는 경우)
    const environmentDir = path.join(__dirname, '..', 'environment');
    if (!fs.existsSync(environmentDir)) {
      fs.mkdirSync(environmentDir, { recursive: true });
      console.log('📁 environment 폴더를 생성했습니다.');
    }

    // 확장 프로그램 목록을 파일에 저장
    const extensionsFilePath = path.join(
      environmentDir,
      'cursor-extensions.txt',
    );
    const fileContent = extensions.join('\n') + '\n';

    fs.writeFileSync(extensionsFilePath, fileContent, 'utf8');

    console.log('✅ 확장 프로그램 목록이 저장되었습니다:', extensionsFilePath);
    console.log('\n📋 저장된 확장 프로그램:');
    extensions.forEach((ext, index) => {
      console.log(`   ${index + 1}. ${ext}`);
    });
  } catch (error) {
    console.error('❌ 스크립트 실행 중 오류가 발생했습니다:', error.message);

    // cursor 명령이 없는 경우 안내 메시지
    if (
      error.message.includes('cursor') &&
      error.message.includes('not found')
    ) {
      console.error('\n💡 해결 방법:');
      console.error('   1. Cursor가 설치되어 있는지 확인하세요.');
      console.error('   2. Cursor가 PATH에 추가되어 있는지 확인하세요.');
      console.error(
        '   3. Cursor에서 "Shell Command: Install cursor command in PATH"를 실행하세요.',
      );
    }

    process.exit(1);
  }
}

// 스크립트 실행
exportExtensions();
