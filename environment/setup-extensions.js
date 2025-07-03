#!/usr/bin/env node

/* eslint-disable @typescript-eslint/no-var-requires */
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

/**
 * Cursor 확장 프로그램을 설치하는 스크립트
 * cursor-extensions.txt 파일에서 확장 프로그램 목록을 읽어와 설치
 */
async function setupExtensions() {
  try {
    // 확장 프로그램 목록 파일 경로
    const extensionsFilePath = path.join(
      __dirname,
      '..',
      'environment',
      'cursor-extensions.txt',
    );

    // 파일 존재 여부 확인
    if (!fs.existsSync(extensionsFilePath)) {
      console.error(
        '❌ cursor-extensions.txt 파일을 찾을 수 없습니다:',
        extensionsFilePath,
      );
      process.exit(1);
    }

    // 파일 내용 읽기
    const fileContent = fs.readFileSync(extensionsFilePath, 'utf8');
    const extensions = fileContent
      .split('\n')
      .map((line) => line.trim())
      .filter((line) => line && !line.startsWith('#')); // 빈 줄과 주석 제거

    if (extensions.length === 0) {
      console.log('⚠️  설치할 확장 프로그램이 없습니다.');
      return;
    }

    console.log(
      `🚀 ${extensions.length}개의 Cursor 확장 프로그램을 설치합니다...\n`,
    );

    // 각 확장 프로그램 설치
    let successCount = 0;
    let failureCount = 0;

    for (const extension of extensions) {
      try {
        console.log(`📦 설치 중: ${extension}`);
        execSync(`cursor --install-extension ${extension}`, {
          stdio: 'pipe',
          timeout: 30000, // 30초 타임아웃
        });
        console.log(`✅ 설치 완료: ${extension}`);
        successCount++;
      } catch (error) {
        console.error(`❌ 설치 실패: ${extension}`);
        console.error(`   오류: ${error.message}`);
        failureCount++;
      }
    }

    console.log('\n📊 설치 결과:');
    console.log(`   성공: ${successCount}개`);
    console.log(`   실패: ${failureCount}개`);

    if (failureCount > 0) {
      console.log('\n⚠️  일부 확장 프로그램 설치에 실패했습니다.');
      console.log('   실패한 확장 프로그램은 수동으로 설치해주세요.');
    } else {
      console.log('\n🎉 모든 확장 프로그램 설치가 완료되었습니다!');
    }
  } catch (error) {
    console.error('❌ 스크립트 실행 중 오류가 발생했습니다:', error.message);
    process.exit(1);
  }
}

// 스크립트 실행
setupExtensions();
